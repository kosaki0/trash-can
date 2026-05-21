from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from generator import MedicalRAGGenerator, rewrite_query
from memory_manager import append_message, compress_memory, normalize_chat_history
from retriever import HybridRetriever, IndexManager


LOGGER = logging.getLogger(__name__)

CSV_PATH = r"C:\Users\psj\Desktop\肿瘤科-10000.csv"
CSV_ENCODING = "gb18030"
STORAGE_DIR = "./rag_storage"
OLLAMA_MODEL = "qwen3:8b"
RERANK_TOP_N = 3
MAX_HISTORY_MESSAGES = 6


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User medical question.")
    chat_history: list[ChatMessage] = Field(default_factory=list)
    clinical_summary: str = ""


class ChatResponse(BaseModel):
    answer: str
    source_documents: list[dict[str, Any]]
    chat_history: list[dict[str, str]]
    clinical_summary: str


class RAGService:
    """Lazy-loaded RAG service for normal and streaming chat."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self.index_manager: IndexManager | None = None
        self.retriever: HybridRetriever | None = None
        self.generator: MedicalRAGGenerator | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            LOGGER.info("Initializing RAG service.")
            self.index_manager = IndexManager(storage_dir=STORAGE_DIR)
            self.retriever = self.index_manager.load_or_build(csv_path=CSV_PATH, encoding=CSV_ENCODING)
            self.generator = MedicalRAGGenerator(ollama_model=OLLAMA_MODEL, top_n=RERANK_TOP_N)
            self._initialized = True
            LOGGER.info("RAG service initialized.")

    def chat(self, query: str, chat_history: list[dict[str, str]], clinical_summary: str) -> ChatResponse:
        self.initialize()
        if self.retriever is None or self.generator is None:
            raise RuntimeError("RAG service failed to initialize.")

        truncated_history = chat_history[-MAX_HISTORY_MESSAGES:]
        standalone_query = rewrite_query(current_query=query, chat_history=truncated_history, model=OLLAMA_MODEL)
        current_history = append_message(truncated_history, "user", query)
        answer, source_documents = self.generator.answer_complex_query(
            query=standalone_query,
            retriever=self.retriever,
            clinical_summary=clinical_summary,
        )
        current_history = append_message(current_history, "assistant", answer)
        current_history, updated_summary = compress_memory(
            chat_history=current_history,
            clinical_summary=clinical_summary,
            model=OLLAMA_MODEL,
        )
        return ChatResponse(
            answer=answer,
            source_documents=[self._serialize_document(document) for document in source_documents],
            chat_history=current_history,
            clinical_summary=updated_summary,
        )

    def stream_chat(
        self,
        query: str,
        chat_history: list[dict[str, str]],
        clinical_summary: str,
    ) -> Iterator[str]:
        self.initialize()
        if self.retriever is None or self.generator is None:
            raise RuntimeError("RAG service failed to initialize.")

        truncated_history = chat_history[-MAX_HISTORY_MESSAGES:]
        yield self._stream_event("status", message="正在理解问题并检索知识库...")
        standalone_query = rewrite_query(current_query=query, chat_history=truncated_history, model=OLLAMA_MODEL)
        current_history = append_message(truncated_history, "user", query)

        yield self._stream_event("status", message="正在重排候选资料...")
        context, source_documents = self.generator.prepare_complex_query_context(
            query=standalone_query,
            retriever=self.retriever,
            clinical_summary=clinical_summary,
        )

        answer_parts: list[str] = []
        yield self._stream_event("start", message="开始生成回答。")
        for token in self.generator.stream_answer(
            query=standalone_query,
            context=context,
            clinical_summary=clinical_summary,
        ):
            answer_parts.append(token)
            yield self._stream_event("token", content=token)

        answer = "".join(answer_parts).strip()
        current_history = append_message(current_history, "assistant", answer)
        current_history, updated_summary = compress_memory(
            chat_history=current_history,
            clinical_summary=clinical_summary,
            model=OLLAMA_MODEL,
        )
        yield self._stream_event(
            "done",
            answer=answer,
            source_documents=[self._serialize_document(document) for document in source_documents],
            chat_history=current_history,
            clinical_summary=updated_summary,
        )

    @staticmethod
    def _stream_event(event_type: str, **payload: Any) -> str:
        payload["type"] = event_type
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @staticmethod
    def _serialize_document(document: dict[str, Any]) -> dict[str, Any]:
        metadata = document.get("metadata", {})
        if is_dataclass(metadata):
            metadata = asdict(metadata)
        return {
            "page_content": str(document.get("page_content", "")),
            "rerank_score": float(document.get("rerank_score", 0.0)),
            "matched_sub_query": str(document.get("matched_sub_query", "")),
            "metadata": dict(metadata),
        }


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logging.getLogger("uvicorn.access").disabled = True

app = FastAPI(title="肿瘤科智能问答专家 API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty.")
    try:
        history = normalize_chat_history([_dump_message(message) for message in request.chat_history])
        return rag_service.chat(
            query=query,
            chat_history=history[-MAX_HISTORY_MESSAGES:],
            clinical_summary=request.clinical_summary.strip(),
        )
    except Exception as exc:
        LOGGER.exception("Failed to process chat query.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty.")
    try:
        history = normalize_chat_history([_dump_message(message) for message in request.chat_history])
        return StreamingResponse(
            rag_service.stream_chat(
                query=query,
                chat_history=history[-MAX_HISTORY_MESSAGES:],
                clinical_summary=request.clinical_summary.strip(),
            ),
            media_type="application/x-ndjson; charset=utf-8",
        )
    except Exception as exc:
        LOGGER.exception("Failed to start streaming chat query.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _dump_message(message: ChatMessage) -> dict[str, str]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
