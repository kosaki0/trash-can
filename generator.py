from __future__ import annotations

import ast
import json
import logging
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

import ollama
import torch
from FlagEmbedding import FlagReranker

from data_processor import Document
from retriever import HybridRetriever


LOGGER = logging.getLogger(__name__)


DocumentInput = Document | Mapping[str, Any]


QUERY_REWRITE_SYSTEM_PROMPT = (
    "你是一个专业的医疗对话意图重写专家。请阅读以下【历史对话】和【用户最新回复】。"
    "用户的最新回复中可能包含了代词（如‘这个病’、‘那个药’、‘它’）或省略了主语。"
    "你的任务是：结合历史对话，将【用户最新回复】重写为一个完整、独立、毫无歧义的医学检索问题。"
    "必须把所有的代词明确替换为具体的疾病名称或医学术语。"
    "请直接输出重写后的独立问题，绝对不要输出任何解释、分析或多余的标点符号。"
)


def rewrite_query(
    current_query: str,
    chat_history: list[dict[str, str]],
    model: str = "qwen3:8b",
) -> str:
    """Rewrite a contextual user query into a standalone medical retrieval query."""
    query = current_query.strip()
    if not query:
        raise ValueError("current_query cannot be empty.")
    if not chat_history:
        return query

    history_text = json.dumps(chat_history, ensure_ascii=False)
    user_prompt = f"【历史对话】：{history_text}\n【用户最新回复】：{query}"
    LOGGER.info("Rewriting query with %d history messages.", len(chat_history))
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        rewritten_query = response["message"]["content"].strip().strip("。！？,.，")
        LOGGER.info("Query rewritten: raw=%r, standalone=%r", query, rewritten_query)
        return rewritten_query or query
    except Exception as exc:
        LOGGER.warning("Query rewrite failed; falling back to raw query. Error: %s", exc)
        return query


class MedicalRAGGenerator:
    """Query decomposition, concurrent retrieval/reranking, and Ollama generation."""

    BASE_SYSTEM_PROMPT = (
        "你是一名严谨、务实的肿瘤科专家。请优先基于提供的 Context 解答用户问题。\n"
        "回答规则：\n"
        "1. Context 由若干条历史问答片段组成，可能不完全逐字匹配用户问题；只要疾病、症状或处理方向相关，"
        "就应综合提炼出有帮助的回答。\n"
        "2. 如果 Context 中包含治疗方式、缓解建议、护理方法、饮食调理、检查或就医建议，请按要点结构化回答。\n"
        "3. 如果 Context 只提供症状描述而没有明确治疗方案，请说明知识库中可确认的信息，并建议及时就医评估，"
        "不要直接拒答。\n"
        "4. 只有当 Context 与用户问题的疾病和症状均明显无关时，才回答："
        "'当前的知识库暂未包含针对该具体问题的相关信息'。\n"
        "5. 不要使用外部网络信息，不要编造具体药物剂量、手术方案或检查结论。\n"
        "6. 面对复合问题时，请分点回答，尽量覆盖用户提出的每个子问题。"
    )

    DECOMPOSE_PROMPT = (
        "你是一个意图分析专家。请判断用户的输入是否包含多个独立的问题。"
        "如果是，请将其拆分为独立的子问题列表；如果是单一问题，请直接输出该问题。"
        "必须且仅输出合法的 Python 列表格式，例如：['问题1', '问题2']。"
    )

    def __init__(
        self,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        ollama_model: str = "qwen3:8b",
        top_n: int = 3,
    ) -> None:
        self.reranker_model_name = reranker_model_name
        self.ollama_model = ollama_model
        self.top_n = top_n
        self._reranker_lock = threading.Lock()

        LOGGER.info("Loading reranker model: %s", self.reranker_model_name)
        self.reranker = FlagReranker(self.reranker_model_name, use_fp16=True)
        self._patch_reranker_tokenizer()
        self._move_reranker_to_gpu_eval()
        LOGGER.info("Generator initialized with Ollama model=%s.", self.ollama_model)

    def decompose_query(self, query: str) -> list[str]:
        """Use Ollama to split a compound question into sub-questions."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        LOGGER.info("Decomposing query with Ollama: %s", query)
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": self.DECOMPOSE_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
            content = response["message"]["content"].strip()
            sub_queries = self._parse_query_list(content)
        except Exception as exc:
            LOGGER.warning("Query decomposition failed; using original query. Error: %s", exc)
            sub_queries = [query]

        LOGGER.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)
        return sub_queries

    def retrieve_and_rerank_concurrently(
        self,
        query: str,
        retriever: HybridRetriever,
        retrieve_top_k: int = 30,
        candidate_k: int = 100,
        rerank_top_n: int = 2,
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval and Top-2 reranking for each sub-query concurrently."""
        sub_queries = self.decompose_query(query)
        all_documents: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(sub_queries)))) as executor:
            futures = {
                executor.submit(
                    self._retrieve_and_rerank_one,
                    sub_query,
                    retriever,
                    retrieve_top_k,
                    candidate_k,
                    rerank_top_n,
                ): sub_query
                for sub_query in sub_queries
            }

            for future in as_completed(futures):
                sub_query = futures[future]
                try:
                    documents = future.result()
                except Exception as exc:
                    LOGGER.exception("Sub-query retrieval failed: %s", sub_query)
                    continue

                for document in documents:
                    metadata = document.get("metadata", {})
                    doc_id = str(metadata.get("doc_id") or metadata.get("retrieval_doc_index") or id(document))
                    existing = all_documents.get(doc_id)
                    if existing is None or float(document.get("rerank_score", 0.0)) > float(existing.get("rerank_score", 0.0)):
                        document["matched_sub_query"] = sub_query
                        all_documents[doc_id] = document

        merged = sorted(all_documents.values(), key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        LOGGER.info("Merged concurrent retrieval pool size=%d.", len(merged))
        return merged

    def rerank_documents(
        self,
        query: str,
        documents: Sequence[DocumentInput],
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Rerank candidate documents by cross-encoder score."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        normalized_docs = [self._normalize_document(document) for document in documents]
        if not normalized_docs:
            LOGGER.warning("No candidate documents were provided for reranking.")
            return []

        pairs = [[query, document["page_content"]] for document in normalized_docs]
        with self._reranker_lock:
            scores = self.reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]

        reranked: list[dict[str, Any]] = []
        for document, score in zip(normalized_docs, scores):
            score_value = float(score)
            reranked.append(
                {
                    **document,
                    "rerank_score": score_value,
                    "metadata": {
                        **document.get("metadata", {}),
                        "rerank_score": score_value,
                    },
                }
            )

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        selected = reranked[:top_n]
        for rank, document in enumerate(selected, start=1):
            metadata = document["metadata"]
            LOGGER.info(
                "Rerank Top-%d: doc_id=%s, department=%s, score=%.6f",
                rank,
                metadata.get("doc_id", ""),
                metadata.get("department", ""),
                document["rerank_score"],
            )
        return selected

    def build_context(self, reranked_documents: Sequence[Mapping[str, Any]]) -> str:
        """Build LLM context from global reranked document pool."""
        chunks: list[str] = []
        for index, document in enumerate(reranked_documents, start=1):
            metadata = dict(document.get("metadata", {}))
            raw_answer_chunk = str(metadata.get("raw_answer_chunk", "")).strip()
            if not raw_answer_chunk:
                continue
            chunks.append(
                "\n".join(
                    [
                        f"[Context {index}]",
                        f"匹配子问题：{document.get('matched_sub_query', '')}",
                        f"疾病/科室：{metadata.get('department', '')}",
                        f"历史患者问题：{metadata.get('combined_query', '')}",
                        f"医生解答：{raw_answer_chunk}",
                    ]
                )
            )

        context = "\n\n".join(chunks) if chunks else "无可用 Context。"
        LOGGER.info("Built prompt Context from %d chunks.", len(chunks))
        return context

    def generate_answer(self, query: str, context: str, clinical_summary: str = "") -> str:
        """Generate final answer with local Ollama service."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        summary_text = clinical_summary.strip() or "暂无患者历史摘要。"
        system_prompt = f"患者历史摘要：{summary_text}\n\n{self.BASE_SYSTEM_PROMPT}"
        user_prompt = f"Context: {context}\n\nUser Query: {query}"
        LOGGER.info("Calling ollama.chat with model=%s.", self.ollama_model)
        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response["message"]["content"].strip()
        LOGGER.info("Final Ollama answer:\n%s", answer)
        return answer

    def answer_complex_query(
        self,
        query: str,
        retriever: HybridRetriever,
        clinical_summary: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Full multi-intent RAG pipeline: decompose, retrieve, rerank, merge, generate."""
        context, top_documents = self.prepare_complex_query_context(
            query=query,
            retriever=retriever,
            clinical_summary=clinical_summary,
        )
        answer = self.generate_answer(query=query, context=context, clinical_summary=clinical_summary)
        return answer, top_documents

    def prepare_complex_query_context(
        self,
        query: str,
        retriever: HybridRetriever,
        clinical_summary: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run retrieval/reranking and build Context before generation."""
        global_documents = self.retrieve_and_rerank_concurrently(query=query, retriever=retriever)
        top_documents = global_documents[: self.top_n]
        context = self.build_context(top_documents)
        return context, top_documents

    def stream_answer(self, query: str, context: str, clinical_summary: str = "") -> Iterator[str]:
        """Stream final answer tokens from local Ollama service."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        summary_text = clinical_summary.strip() or "暂无患者历史摘要。"
        system_prompt = f"患者历史摘要：{summary_text}\n\n{self.BASE_SYSTEM_PROMPT}"
        user_prompt = f"Context: {context}\n\nUser Query: {query}"
        LOGGER.info("Calling streaming ollama.chat with model=%s.", self.ollama_model)
        stream = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    def answer_with_documents(
        self,
        query: str,
        documents: Sequence[DocumentInput],
        clinical_summary: str = "",
    ) -> str:
        """Compatibility path for callers that already have retrieved documents."""
        reranked_documents = self.rerank_documents(query=query, documents=documents, top_n=self.top_n)
        context = self.build_context(reranked_documents)
        return self.generate_answer(query=query, context=context, clinical_summary=clinical_summary)

    def _retrieve_and_rerank_one(
        self,
        sub_query: str,
        retriever: HybridRetriever,
        retrieve_top_k: int,
        candidate_k: int,
        rerank_top_n: int,
    ) -> list[dict[str, Any]]:
        retrieved = retriever.retrieve(query=sub_query, top_k=retrieve_top_k, candidate_k=candidate_k)
        return self.rerank_documents(query=sub_query, documents=retrieved, top_n=rerank_top_n)

    def _move_reranker_to_gpu_eval(self) -> None:
        model = getattr(self.reranker, "model", None)
        if model is None:
            LOGGER.warning("Reranker model attribute not found; skipping explicit eval/cuda.")
            return

        if torch.cuda.is_available():
            model.cuda()
        else:
            LOGGER.warning("CUDA is unavailable; reranker will use default device.")
        model.eval()

    def _patch_reranker_tokenizer(self) -> None:
        tokenizer = getattr(self.reranker, "tokenizer", None)
        if tokenizer is None or hasattr(tokenizer, "prepare_for_model"):
            return

        LOGGER.warning("Applying reranker tokenizer compatibility patch.")

        def prepare_for_model(
            ids: list[int],
            pair_ids: list[int] | None = None,
            truncation: str | bool = "only_second",
            max_length: int | None = None,
            padding: bool = False,
            **_: Any,
        ) -> dict[str, list[int]]:
            first_ids = list(ids)
            second_ids = list(pair_ids or [])
            if max_length is not None:
                special_tokens = 4 if pair_ids is not None else 2
                available_length = max(0, max_length - special_tokens)
                if pair_ids is not None and truncation == "only_second":
                    second_ids = second_ids[: max(0, available_length - len(first_ids))]
                elif pair_ids is not None:
                    first_available = available_length // 2
                    first_ids = first_ids[:first_available]
                    second_ids = second_ids[: max(0, available_length - len(first_ids))]
                else:
                    first_ids = first_ids[:available_length]

            cls_token_id = tokenizer.cls_token_id
            sep_token_id = tokenizer.sep_token_id
            if cls_token_id is None or sep_token_id is None:
                raise ValueError("Reranker tokenizer must provide cls_token_id and sep_token_id.")

            if pair_ids is None:
                input_ids = [cls_token_id, *first_ids, sep_token_id]
            else:
                input_ids = [cls_token_id, *first_ids, sep_token_id, sep_token_id, *second_ids, sep_token_id]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        tokenizer.prepare_for_model = prepare_for_model

    @classmethod
    def _parse_query_list(cls, text: str) -> list[str]:
        match = re.search(r"\[[\s\S]*\]", text)
        candidate = match.group(0) if match else text
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, str):
            queries = [parsed]
        elif isinstance(parsed, list):
            queries = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            raise ValueError(f"Unexpected decomposition result: {text}")
        return queries or [text.strip()]

    @staticmethod
    def _normalize_document(document: DocumentInput) -> dict[str, Any]:
        if isinstance(document, Mapping):
            return {
                "page_content": str(document.get("page_content", "")),
                "metadata": dict(document.get("metadata", {})),
            }
        return {"page_content": document.page_content, "metadata": dict(document.metadata)}
