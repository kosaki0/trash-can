from __future__ import annotations

import argparse
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

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
    """Evaluate hybrid retrieval and reranking quality with doc_id ground truth."""

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
        """Run two-stage evaluation: hybrid Top-30 and reranked Top-3."""
        test_documents = self._sample_test_documents()
        stage_one_ranks: list[int | None] = []
        stage_two_ranks: list[int | None] = []

        for index, test_document in enumerate(test_documents, start=1):
            query = str(test_document.metadata["combined_query"])
            ground_truth_doc_id = str(test_document.metadata["doc_id"])

            retrieved_docs = self.retriever.retrieve(query, top_k=30, candidate_k=100)
            stage_one_ranks.append(self._find_rank_by_doc_id(retrieved_docs, ground_truth_doc_id, cutoff=30))

            reranked_docs = self.generator.rerank_documents(query, retrieved_docs, top_n=3)
            stage_two_ranks.append(self._find_rank_by_doc_id(reranked_docs, ground_truth_doc_id, cutoff=3))

            LOGGER.info(
                "Evaluated sample %d/%d: doc_id=%s, stage1_rank=%s, stage2_rank=%s",
                index,
                len(test_documents),
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

    def _sample_test_documents(self) -> list[Document]:
        documents = self._load_documents()
        if not documents:
            raise ValueError("documents.pkl contains no documents.")

        sample_count = min(self.sample_size, len(documents))
        random.seed(self.random_seed)
        sampled = random.sample(documents, sample_count)
        LOGGER.info("Sampled %d documents for evaluation.", len(sampled))
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
        LOGGER.info("========== RAG Evaluation Report ==========")
        LOGGER.info("Hybrid Retrieval Top-30: Hit Rate@30=%.4f, MRR@30=%.4f, Total=%d", hybrid.hit_rate, hybrid.mrr, hybrid.total)
        LOGGER.info("Reranker Top-3: Hit Rate@3=%.4f, MRR@3=%.4f, Total=%d", rerank.hit_rate, rerank.mrr, rerank.total)
        LOGGER.info("===========================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oncology RAG retrieval and reranking.")
    parser.add_argument(
        "--csv-path",
        default=r"C:\Users\psj\Desktop\肿瘤科5-10000.csv",
        help="Path to the oncology Q&A CSV file. Used when persistent indexes do not exist.",
    )
    parser.add_argument("--encoding", default="gb18030", help="CSV encoding.")
    parser.add_argument("--storage-dir", default="./rag_storage", help="Directory for persistent indexes.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of evaluation samples.")
    parser.add_argument("--ollama-model", default="qwen3:8b", help="Local Ollama model name; not used during rerank metrics.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    index_manager = IndexManager(storage_dir=args.storage_dir)
    retriever = index_manager.load_or_build(csv_path=args.csv_path, encoding=args.encoding)
    generator = MedicalRAGGenerator(ollama_model=args.ollama_model)
    evaluator = MedicalRAGEvaluator(
        index_manager=index_manager,
        retriever=retriever,
        generator=generator,
        sample_size=args.sample_size,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
