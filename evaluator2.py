from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import ollama

from data_processor import Document
from generator import MedicalRAGGenerator
from retriever import HybridRetriever, IndexManager

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationMetrics:
    hit_rate: float
    mrr: float
    total: int


class MedicalRAGEvaluator:
    """Evaluate RAG quality and mine Hard Negative triplets with robust alignment guardrails."""

    def __init__(
            self,
            index_manager: IndexManager,
            retriever: HybridRetriever,
            generator: MedicalRAGGenerator,
            sample_size: int = 100,
            random_seed: int = 42,
    ) -> None:
        self.index_manager = index_manager
        self.retriever = retriever
        self.generator = generator
        self.sample_size = sample_size
        self.random_seed = random_seed

    def evaluate(self) -> dict[str, EvaluationMetrics]:
        """Run evaluation using aligned colloquial patient queries."""
        test_documents = self._sample_test_documents()
        stage_one_ranks: list[int | None] = []
        stage_two_ranks: list[int | None] = []

        for index, test_document in enumerate(test_documents, start=1):
            original_query = str(test_document.metadata["combined_query"])
            ground_truth_doc_id = str(test_document.metadata["doc_id"])

            # 生成经过严格约束的口语化提问
            patient_query = self._generate_patient_query(original_query)

            LOGGER.info("Sample %d/%d", index, len(test_documents))
            LOGGER.info("  Original Query : %s", original_query)
            LOGGER.info("  Patient Query  : %s", patient_query)

            # 使用口语化提问进行混合检索
            retrieved_docs = self.retriever.retrieve(patient_query, top_k=30, candidate_k=100)
            stage_one_ranks.append(self._find_rank_by_doc_id(retrieved_docs, ground_truth_doc_id, cutoff=30))

            # 使用口语化提问进行 Rerank
            reranked_docs = self.generator.rerank_documents(patient_query, retrieved_docs, top_n=3)
            stage_two_ranks.append(self._find_rank_by_doc_id(reranked_docs, ground_truth_doc_id, cutoff=3))

            LOGGER.info(
                "  Results -> doc_id: %s | stage1_rank: %s | stage2_rank: %s\n",
                ground_truth_doc_id,
                stage_one_ranks[-1],
                stage_two_ranks[-1],
            )

        report = {
            "hybrid_top_30": self._calculate_metrics(stage_one_ranks),
            "rerank_top_3": self._calculate_metrics(stage_two_ranks),
        }
        self._log_report(report)
        return report

    def mine_training_triplets(self, output_path: str = "training_triplets.jsonl", num_negatives: int = 5) -> None:
        """Mine (Query, Positive, Hard Negative) triplets for fine-tuning BGE embeddings."""
        documents = self._load_documents()
        doc_store = {str(doc.metadata.get("doc_id", "")): doc for doc in documents}

        test_documents = self._sample_test_documents()
        output_file = Path(output_path)
        mined_count = 0

        with output_file.open("w", encoding="utf-8") as f:
            for index, test_document in enumerate(test_documents, start=1):
                original_query = str(test_document.metadata["combined_query"])
                truth_id = str(test_document.metadata["doc_id"])

                if truth_id not in doc_store:
                    continue

                # 1. 生成口语化 Query
                patient_query = self._generate_patient_query(original_query)
                positive_doc_content = doc_store[truth_id].page_content

                # 2. 挖掘 Hard Negatives (直接复用原 retriever 里的私有 BM25 检索逻辑)
                sparse_hits = self.retriever._sparse_retrieve(patient_query, top_k=20, allowed_domains=None)

                hard_negatives = []
                for hit in sparse_hits:
                    doc_id = str(self.retriever.documents[hit.doc_index].metadata.get("doc_id", ""))
                    if doc_id != truth_id:
                        content = self.retriever.documents[hit.doc_index].page_content
                        if content not in hard_negatives:
                            hard_negatives.append(content)

                # 3. 如果硬负样本数量不足，随机抽取填充
                while len(hard_negatives) < num_negatives:
                    rand_doc = random.choice(documents)
                    rand_id = str(rand_doc.metadata.get("doc_id", ""))
                    if rand_id != truth_id and rand_doc.page_content not in hard_negatives:
                        hard_negatives.append(rand_doc.page_content)

                selected_negatives = hard_negatives[:num_negatives]

                # 4. 写入三元组数据文件
                triplet = {
                    "query": patient_query,
                    "pos": [positive_doc_content],
                    "neg": selected_negatives
                }

                f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                mined_count += 1

                if index % 10 == 0 or index == len(test_documents):
                    LOGGER.info("Progress: Mined %d/%d triplets...", index, len(test_documents))

        LOGGER.info("Successfully mined %d pure triplets to %s", mined_count, output_path)

    def _generate_patient_query(self, original_query: str) -> str:
        """Generate high-quality oncology patient queries with strict guardrails."""
        prompt_template = (
            "【角色锚定】：你现在是一名正在接受治疗的成人肿瘤/癌症患者，或者是该成人患者的家属。\n"
            "【核心任务】：将给定的医学问题，改写为一句日常、口语化的网络求助或面诊提问。\n"
            "【绝对禁忌 - 违背将导致系统崩溃】：\n"
            "1. 严禁出现“娃”、“宝宝”、“孩子”、“小儿”、“孕妇”！必须是成人的日常真实场景！\n"
            "2. 严禁发散到“普通感冒”、“日常发热”、“拉肚子”、“骨折”、“拔牙”等非肿瘤小病！\n"
            "3. 如果原文只有“发热”或“呕吐”等通用症状，你必须主动加上“化疗后”、“吃靶向药”、“癌症晚期”、“放疗术后”等肿瘤特定背景词汇。\n\n"
            "【改写示例】：\n"
            "原文：化疗引起的恶心呕吐如何缓解？ -> 提问：我爸这几天刚做完化疗，一直吐个不停，吃什么都吃不下，该怎么办啊？\n"
            "原文：发热的鉴别诊断 -> 提问：医生，我妈肺癌晚期，这两天突然开始发低烧，这正常吗？\n"
            "原文：基因突变检测的意义 -> 提问：大夫让我去做什么靶向基因检测，这玩意儿到底有啥用，必须得做吗？\n\n"
            "【当前任务】：\n"
            "原文：{standard_text}\n"
            "肿瘤患者的口语化提问："
        )
        try:
            response = ollama.chat(
                model=self.generator.ollama_model,
                messages=[{"role": "user", "content": prompt_template.format(standard_text=original_query)}],
                options={"temperature": 0.2, "top_p": 0.8}  # 极低温度，全面抑制大模型的发散幻觉
            )
            patient_query = response["message"]["content"].strip().strip('"\'')

            # ----------------------------------------------------
            # 护栏防线一：后置敏感词过滤器 (拦截儿科/感冒等幻觉)
            # ----------------------------------------------------
            forbidden_words = ["娃", "宝宝", "小孩", "儿科", "感冒", "摔伤", "拔牙", "牙痛", "孕妇", "小儿", "孩子"]
            if any(word in patient_query for word in forbidden_words):
                LOGGER.warning("触发安全护栏（命中非肿瘤敏感词）: %r -> 降级回退使用原 Query", patient_query)
                return original_query

            # ----------------------------------------------------
            # 护栏防线二：肿瘤特征弱匹配校验
            # ----------------------------------------------------
            oncology_keywords = ["癌", "瘤", "化疗", "放疗", "靶向", "手术", "转移", "复发", "结节", "指标", "医生",
                                 "大夫", "医院", "吐", "烧", "药"]
            if not any(word in patient_query for word in oncology_keywords) and len(patient_query) < 12:
                LOGGER.warning("触发安全护栏（缺乏肿瘤语境特征）: %r -> 降级回退使用原 Query", patient_query)
                return original_query

            return patient_query

        except Exception as e:
            LOGGER.warning("Ollama 提问改写失败，错误: %s -> 回退使用原 Query", e)
            return original_query

    def _sample_test_documents(self) -> list[Document]:
        documents = self._load_documents()
        if not documents:
            raise ValueError("documents.pkl contains no documents.")

        sample_count = min(self.sample_size, len(documents))
        random.seed(self.random_seed)
        sampled = random.sample(documents, sample_count)
        LOGGER.info("Sampled %d documents for run.", len(sampled))
        return sampled

    def _load_documents(self) -> list[Document]:
        documents_path = self.index_manager.documents_path
        if not documents_path.exists():
            raise FileNotFoundError(f"documents.pkl does not exist: {documents_path}")

        with documents_path.open("rb") as file:
            documents: list[Document] = pickle.load(file)
        return documents

    @staticmethod
    def _find_rank_by_doc_id(
            documents: Sequence[Document] | Sequence[dict[str, object]],
            ground_truth_doc_id: str,
            cutoff: int,
    ) -> int | None:
        for rank, document in enumerate(documents[:cutoff], start=1):
            metadata = document["metadata"] if isinstance(document, dict) else document.metadata
            if str(metadata.get("doc_id", "")) == ground_truth_doc_id:
                return rank
        return None

    @staticmethod
    def _calculate_metrics(ranks: Sequence[int | None]) -> EvaluationMetrics:
        total = len(ranks)
        if total == 0:
            return EvaluationMetrics(hit_rate=0.0, mrr=0.0, total=0)

        hits = [rank for rank in ranks if rank is not None]
        hit_rate = len(hits) / total
        mrr = sum(1.0 / rank for rank in hits if rank is not None) / total
        return EvaluationMetrics(hit_rate=hit_rate, mrr=mrr, total=total)

    @staticmethod
    def _log_report(report: dict[str, EvaluationMetrics]) -> None:
        hybrid = report["hybrid_top_30"]
        rerank = report["rerank_top_3"]
        LOGGER.info("========== RAG Evaluation Report (Aligned) ==========")
        LOGGER.info("Hybrid Retrieval Top-30: Hit Rate@30=%.4f, MRR@30=%.4f, Total=%d", hybrid.hit_rate, hybrid.mrr,
                    hybrid.total)
        LOGGER.info("Reranker Top-3:          Hit Rate@3=%.4f,  MRR@3=%.4f,  Total=%d", rerank.hit_rate, rerank.mrr,
                    rerank.total)
        LOGGER.info("=====================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oncology RAG or Mine Triplets with Guardrails.")
    parser.add_argument("--mode", choices=["eval", "mine"], default="eval", help="Run evaluation or mine triplets.")
    parser.add_argument("--triplets-out", default="training_triplets.jsonl", help="Output path for mined triplets.")
    parser.add_argument("--csv-path", default=r"C:\Users\psj\Desktop\肿瘤科5-10000.csv",
                        help="Path to the oncology Q&A CSV.")
    parser.add_argument("--encoding", default="gb18030", help="CSV encoding.")
    parser.add_argument("--storage-dir", default="./rag_storage", help="Directory for persistent indexes.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to process.")
    parser.add_argument("--ollama-model", default="qwen3:8b", help="Local Ollama model name.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    index_manager = IndexManager(storage_dir=args.storage_dir)
    retriever = index_manager.load_or_build(csv_path=args.csv_path, encoding=args.encoding)
    generator = MedicalRAGGenerator(ollama_model=args.ollama_model)

    evaluator = MedicalRAGEvaluator(
        index_manager=index_manager,
        retriever=retriever,
        generator=generator,
        sample_size=args.sample_size,
    )

    if args.mode == "eval":
        evaluator.evaluate()
    elif args.mode == "mine":
        evaluator.mine_training_triplets(output_path=args.triplets_out)


if __name__ == "__main__":
    main()