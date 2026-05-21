from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from data_processor import Document, MedicalDataProcessor


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalHit:
    doc_index: int
    score: float
    rank: int


class IndexManager:
    """Persist and load FAISS, BM25, and Documents for fast online startup."""

    FAISS_FILE: Final[str] = "faiss_index.bin"
    BM25_FILE: Final[str] = "bm25_model.pkl"
    DOCUMENTS_FILE: Final[str] = "documents.pkl"

    def __init__(
        self,
        storage_dir: str | Path = "./rag_storage",
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        normalize_embeddings: bool = True,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.embedding_model_name = embedding_model_name
        self.normalize_embeddings = normalize_embeddings
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model: SentenceTransformer | None = None

    @property
    def faiss_path(self) -> Path:
        return self.storage_dir / self.FAISS_FILE

    @property
    def bm25_path(self) -> Path:
        return self.storage_dir / self.BM25_FILE

    @property
    def documents_path(self) -> Path:
        return self.storage_dir / self.DOCUMENTS_FILE

    def index_exists(self) -> bool:
        return self.faiss_path.exists() and self.bm25_path.exists() and self.documents_path.exists()

    def build_from_csv(self, csv_path: str | Path, encoding: str = "gb18030") -> HybridRetriever:
        """Offline indexing: parse CSV, build documents, compute embeddings, persist indexes."""
        LOGGER.info("Building persistent RAG indexes from CSV: %s", csv_path)
        processor = MedicalDataProcessor(csv_path=csv_path, encoding=encoding)
        documents = processor.build_documents()
        faiss_index = self._build_faiss_index(documents)
        bm25_model = self._build_bm25_model(documents)

        faiss.write_index(faiss_index, str(self.faiss_path))
        with self.bm25_path.open("wb") as file:
            pickle.dump(bm25_model, file)
        with self.documents_path.open("wb") as file:
            pickle.dump(documents, file)

        LOGGER.info("Saved FAISS, BM25, and Documents to %s", self.storage_dir)
        return HybridRetriever(
            documents=documents,
            faiss_index=faiss_index,
            bm25_model=bm25_model,
            embedding_model=self._get_embedding_model(),
            normalize_embeddings=self.normalize_embeddings,
        )

    def load(self) -> HybridRetriever:
        """Online loading: read persisted indexes and skip CSV parsing/embedding computation."""
        if not self.index_exists():
            raise FileNotFoundError(f"Persistent indexes are incomplete under {self.storage_dir}")

        LOGGER.info("Loading persistent indexes from %s", self.storage_dir)
        faiss_index = faiss.read_index(str(self.faiss_path))
        with self.bm25_path.open("rb") as file:
            bm25_model: BM25Okapi = pickle.load(file)
        with self.documents_path.open("rb") as file:
            documents: list[Document] = pickle.load(file)
        for document in documents:
            document.metadata.setdefault("domain", "oncology")

        LOGGER.info("Loaded %d documents from persistent storage.", len(documents))
        return HybridRetriever(
            documents=documents,
            faiss_index=faiss_index,
            bm25_model=bm25_model,
            embedding_model=self._get_embedding_model(),
            normalize_embeddings=self.normalize_embeddings,
        )

    def load_or_build(self, csv_path: str | Path, encoding: str = "gb18030") -> HybridRetriever:
        if self.index_exists():
            return self.load()
        return self.build_from_csv(csv_path=csv_path, encoding=encoding)

    def persist(self, retriever: HybridRetriever) -> None:
        """Persist a retriever after online document ingestion."""
        faiss.write_index(retriever.faiss_index, str(self.faiss_path))
        with self.bm25_path.open("wb") as file:
            pickle.dump(retriever.bm25_model, file)
        with self.documents_path.open("wb") as file:
            pickle.dump(retriever.documents, file)
        LOGGER.info("Persisted updated indexes to %s.", self.storage_dir)

    def rebuild_faiss_index(self, documents: Sequence[Document]) -> faiss.Index:
        return self._build_faiss_index(documents)

    def _build_faiss_index(self, documents: Sequence[Document]) -> faiss.Index:
        texts = [document.page_content for document in documents]
        LOGGER.info("Encoding %d documents with %s", len(texts), self.embedding_model_name)
        embeddings = self._get_embedding_model().encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        LOGGER.info("Built FAISS index: ntotal=%d, dim=%d", index.ntotal, embeddings.shape[1])
        return index

    @staticmethod
    def _build_bm25_model(documents: Sequence[Document]) -> BM25Okapi:
        tokenized_corpus = [HybridRetriever.tokenize(document.page_content) for document in documents]
        LOGGER.info("Built BM25 model for %d documents.", len(tokenized_corpus))
        return BM25Okapi(tokenized_corpus)

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model is None:
            LOGGER.info("Loading embedding model: %s", self.embedding_model_name)
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model


class HybridRetriever:
    """FAISS + BM25 hybrid retriever with RRF fusion."""

    def __init__(
        self,
        documents: Sequence[Document],
        faiss_index: faiss.Index,
        bm25_model: BM25Okapi,
        embedding_model: SentenceTransformer,
        normalize_embeddings: bool = True,
        rrf_k: int = 60,
    ) -> None:
        self.documents = list(documents)
        self.faiss_index = faiss_index
        self.bm25_model = bm25_model
        self.embedding_model = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.rrf_k = rrf_k
        self._dense_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def retrieve(self, query: str, top_k: int = 30, candidate_k: int = 100) -> list[Document]:
        """Retrieve Top-30 by default after dense/sparse Top-100 recall and RRF fusion."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        allowed_domains = self._allowed_domains_for_query(query)
        with self._write_lock:
            dense_hits = self._dense_retrieve(query, candidate_k, allowed_domains=allowed_domains)
            sparse_hits = self._sparse_retrieve(query, candidate_k, allowed_domains=allowed_domains)
            fused = self._rrf_fuse(dense_hits=dense_hits, sparse_hits=sparse_hits, top_k=top_k)
        LOGGER.info(
            "Hybrid retrieve query=%r domains=%s: dense=%d, sparse=%d, fused=%d",
            query,
            sorted(allowed_domains) if allowed_domains else "all",
            len(dense_hits),
            len(sparse_hits),
            len(fused),
        )
        return [self._copy_document_with_metadata(doc_index, rank, score) for rank, (doc_index, score) in enumerate(fused, start=1)]

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Append uploaded documents to FAISS and rebuild BM25."""
        new_documents = list(documents)
        if not new_documents:
            return

        with self._write_lock:
            texts = [document.page_content for document in new_documents]
            LOGGER.info("Encoding %d uploaded document chunks.", len(texts))
            with self._dense_lock:
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            LOGGER.info("Adding uploaded embeddings to FAISS index.")
            self.faiss_index.add(embeddings)
            self.documents.extend(new_documents)
            LOGGER.info("Rebuilding BM25 model for %d total documents.", len(self.documents))
            self.bm25_model = BM25Okapi([self.tokenize(document.page_content) for document in self.documents])
        LOGGER.info("Added %d uploaded documents; corpus size=%d.", len(new_documents), len(self.documents))

    def remove_uploaded_batch(self, upload_batch_id: str) -> int:
        """Remove one uploaded batch and rebuild in-memory FAISS/BM25 indexes."""
        with self._write_lock:
            before_count = len(self.documents)
            self.documents = [
                document
                for document in self.documents
                if document.metadata.get("upload_batch_id") != upload_batch_id
            ]
            removed_count = before_count - len(self.documents)
            if removed_count == 0:
                return 0

            texts = [document.page_content for document in self.documents]
            with self._dense_lock:
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.faiss_index = index
            self.bm25_model = BM25Okapi([self.tokenize(document.page_content) for document in self.documents])
        LOGGER.info("Removed uploaded batch %s: %d documents.", upload_batch_id, removed_count)
        return removed_count

    def list_uploaded_batches(self) -> list[dict[str, object]]:
        batches: dict[str, dict[str, object]] = {}
        for document in self.documents:
            metadata = document.metadata
            batch_id = metadata.get("upload_batch_id")
            if not batch_id:
                continue
            batch_key = str(batch_id)
            item = batches.setdefault(
                batch_key,
                {
                    "batch_id": batch_key,
                    "filename": str(metadata.get("source_file", "")),
                    "label": str(metadata.get("gatekeeper_label", "")),
                    "domain": str(metadata.get("domain", "")),
                    "document_count": 0,
                },
            )
            item["document_count"] = int(item["document_count"]) + 1
        return sorted(batches.values(), key=lambda item: str(item["batch_id"]), reverse=True)

    def _dense_retrieve(self, query: str, top_k: int, allowed_domains: set[str] | None) -> list[RetrievalHit]:
        effective_top_k = min(self._expanded_top_k(top_k, allowed_domains), len(self.documents))
        with self._dense_lock:
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            query_embedding = np.asarray(query_embedding, dtype=np.float32)
            scores, indices = self.faiss_index.search(query_embedding, effective_top_k)
        hits: list[RetrievalHit] = []
        for rank, (doc_index, score) in enumerate(zip(indices[0], scores[0]), start=1):
            doc_index = int(doc_index)
            if doc_index >= 0 and self._document_allowed(doc_index, allowed_domains):
                hits.append(RetrievalHit(doc_index=doc_index, score=float(score), rank=rank))
                if len(hits) >= top_k:
                    break
        return hits

    def _sparse_retrieve(self, query: str, top_k: int, allowed_domains: set[str] | None) -> list[RetrievalHit]:
        effective_top_k = min(top_k, len(self.documents))
        scores = self.bm25_model.get_scores(self.tokenize(query))
        ranked_indices = np.argsort(scores)[::-1]
        hits: list[RetrievalHit] = []
        rank = 1
        for doc_index in ranked_indices:
            doc_index = int(doc_index)
            if not self._document_allowed(doc_index, allowed_domains):
                continue
            hits.append(RetrievalHit(doc_index=doc_index, score=float(scores[doc_index]), rank=rank))
            rank += 1
            if len(hits) >= effective_top_k:
                break
        return hits

    def _rrf_fuse(
        self,
        dense_hits: Sequence[RetrievalHit],
        sparse_hits: Sequence[RetrievalHit],
        top_k: int,
    ) -> list[tuple[int, float]]:
        rrf_scores: dict[int, float] = {}
        for hit in dense_hits:
            rrf_scores[hit.doc_index] = rrf_scores.get(hit.doc_index, 0.0) + 1.0 / (self.rrf_k + hit.rank)
        for hit in sparse_hits:
            rrf_scores[hit.doc_index] = rrf_scores.get(hit.doc_index, 0.0) + 1.0 / (self.rrf_k + hit.rank)

        return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def _copy_document_with_metadata(self, doc_index: int, rank: int, rrf_score: float) -> Document:
        source = self.documents[doc_index]
        metadata = dict(source.metadata)
        metadata.setdefault("domain", "oncology")
        metadata.update(
            {
                "retrieval_rank": rank,
                "retrieval_rrf_score": rrf_score,
                "retrieval_doc_index": doc_index,
            }
        )
        return Document(page_content=source.page_content, metadata=metadata)

    def _allowed_domains_for_query(self, query: str) -> set[str] | None:
        return None if self._has_other_medical_intent(query) else {"oncology"}

    @staticmethod
    def _has_other_medical_intent(query: str) -> bool:
        other_medical_terms = (
            "骨科",
            "牙科",
            "口腔",
            "儿科",
            "皮肤科",
            "眼科",
            "耳鼻喉",
            "妇产科",
            "感冒",
            "发烧",
            "骨折",
            "牙痛",
            "鼻炎",
            "肺炎",
            "高血压",
            "糖尿病",
            "其他科",
            "非肿瘤",
        )
        return any(term in query for term in other_medical_terms)

    def _expanded_top_k(self, top_k: int, allowed_domains: set[str] | None) -> int:
        if allowed_domains is None:
            return top_k
        return min(len(self.documents), max(top_k * 5, top_k))

    def _document_allowed(self, doc_index: int, allowed_domains: set[str] | None) -> bool:
        if allowed_domains is None:
            return True
        domain = str(self.documents[doc_index].metadata.get("domain", "oncology"))
        return domain in allowed_domains

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
