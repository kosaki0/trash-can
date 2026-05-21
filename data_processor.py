from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

import ollama
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(slots=True)
class Document:
    """Standard RAG document object."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MedicalDataProcessor:
    """Clean oncology Q&A data and build structured retrieval documents."""

    REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("department", "title", "ask", "answer")
    DEFAULT_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", "。", "！", "？", "，", "")
    HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")

    def __init__(
        self,
        csv_path: str | Path,
        encoding: str = "gb18030",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data: pd.DataFrame = pd.DataFrame()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=list(self.DEFAULT_SEPARATORS),
            length_function=len,
        )

    def load_data(self) -> pd.DataFrame:
        """Load CSV and validate required fields."""
        LOGGER.info("Loading CSV data from %s", self.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path, encoding=self.encoding)
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        LOGGER.info("Loaded %d rows and %d columns.", len(self.data), len(self.data.columns))
        return self.data

    def process(self) -> pd.DataFrame:
        """Run cleaning, ID injection, and feature fusion."""
        if self.data.empty:
            self.load_data()

        self._inject_doc_id()
        self._build_combined_query()
        self._clean_answer_column()
        return self.data

    def build_documents(self) -> list[Document]:
        """Build structured retrieval documents from processed data."""
        if self.data.empty or "combined_query" not in self.data.columns:
            self.process()

        documents: list[Document] = []
        for record in self._iter_records():
            clean_answer = self._normalize_text(record.get("answer", ""))
            chunks = [clean_answer] if len(clean_answer) <= self.chunk_size else self.text_splitter.split_text(clean_answer)

            for chunk_index, chunked_answer in enumerate(chunks):
                metadata = {
                    "doc_id": str(record["doc_id"]),
                    "domain": "oncology",
                    "department": self._normalize_text(record.get("department", "")),
                    "combined_query": self._normalize_text(record.get("combined_query", "")),
                    "raw_answer_chunk": chunked_answer,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
                page_content = f"【患者问题】：{metadata['combined_query']}\n【医生解答】：{chunked_answer}"
                documents.append(Document(page_content=page_content, metadata=metadata))

        LOGGER.info("Built %d structured Documents from %d rows.", len(documents), len(self.data))
        if documents:
            LOGGER.info("Structured page_content example:\n%s", documents[0].page_content)
        return documents

    def _inject_doc_id(self) -> None:
        if "doc_id" in self.data.columns:
            self.data["doc_id"] = self.data["doc_id"].astype(str)
            LOGGER.info("Using existing doc_id column.")
            return

        self.data.insert(0, "doc_id", [f"doc_{index:06d}" for index in range(len(self.data))])
        LOGGER.info("Injected doc_id for %d rows.", len(self.data))

    def _build_combined_query(self) -> None:
        missing_ask_mask = self._missing_ask_mask()
        title_text = self.data["title"].apply(self._clean_text)
        ask_text = self.data["ask"].apply(self._clean_text)
        self.data["combined_query"] = title_text.where(missing_ask_mask, title_text + "：" + ask_text)
        LOGGER.info("Created combined_query column.")

    def _clean_answer_column(self) -> None:
        self.data["answer"] = self.data["answer"].apply(self._clean_text)
        self.data["answer_length"] = self.data["answer"].str.len().astype("int64")
        LOGGER.info("Cleaned answer column and created answer_length.")

    def _missing_ask_mask(self) -> pd.Series:
        ask_text = self.data["ask"].astype("string").str.strip()
        return self.data["ask"].isna() | ask_text.eq("") | ask_text.eq("无")

    def _iter_records(self) -> Iterable[dict[str, Any]]:
        yield from self.data.to_dict(orient="records")

    @classmethod
    def _clean_text(cls, value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        text = cls.HTML_TAG_PATTERN.sub(" ", text)
        text = cls.WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()

    @staticmethod
    def _normalize_text(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()


class DataGatekeeper:
    """Strict LLM data auditor for uploaded medical knowledge files."""

    SYSTEM_PROMPT: Final[str] = (
        "你是一个极其严格的医疗数据分类引擎。请阅读以下数据样本，并将其严格归入以下三个类别之一：\n"
        "1. [ONCOLOGY]：内容明确属于肿瘤科、癌症治疗、靶向药等相关领域。\n"
        "2. [OTHER_MED]：内容属于医学领域，但属于其他科室（如骨科、牙科、儿科、感冒发烧等），非肿瘤重症。\n"
        "3. [REJECT]：内容与医学毫无关系（如菜谱、小说、财务报表等）。\n\n"
        "请仅输出方括号内的标签名称（如 [ONCOLOGY]），绝对不要输出任何其他解释字符。数据样本：{sample_text}"
    )
    VALID_LABELS: Final[set[str]] = {"[ONCOLOGY]", "[OTHER_MED]", "[REJECT]"}
    DOMAIN_BY_LABEL: Final[dict[str, str]] = {
        "[ONCOLOGY]": "oncology",
        "[OTHER_MED]": "other_medical",
    }

    def __init__(
        self,
        model: str = "qwen3:8b",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )

    def classify(self, text_blocks: list[str]) -> str:
        clean_blocks = [self._clean_text(block) for block in text_blocks if self._clean_text(block)]
        if not clean_blocks:
            raise ValueError("上传文件未解析出可入库文本。")

        sample_text = self._sample_text(clean_blocks)
        prompt = self.SYSTEM_PROMPT.format(sample_text=sample_text)
        LOGGER.info("Auditing uploaded document sample with %s.", self.model)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
        )
        label = self._extract_label(response["message"]["content"])
        LOGGER.info("DataGatekeeper classified upload as %s.", label)
        return label

    def build_documents(
        self,
        text_blocks: list[str],
        label: str,
        source_name: str,
        upload_batch_id: str | None = None,
    ) -> list[Document]:
        if label == "[REJECT]":
            raise ValueError("检测到非医学数据，拒绝污染知识库。")
        if label not in self.DOMAIN_BY_LABEL:
            raise ValueError(f"Unsupported gatekeeper label: {label}")

        domain = self.DOMAIN_BY_LABEL[label]
        documents: list[Document] = []
        source_stem = Path(source_name).stem or "upload"
        clean_blocks = [self._clean_text(block) for block in text_blocks if self._clean_text(block)]

        for block_index, block in enumerate(clean_blocks):
            chunks = [block] if len(block) <= self.chunk_size else self.text_splitter.split_text(block)
            for chunk_index, chunk in enumerate(chunks):
                doc_id = f"upload_{source_stem}_{block_index:05d}_{chunk_index:03d}"
                metadata = {
                    "doc_id": doc_id,
                    "domain": domain,
                    "department": "肿瘤科" if domain == "oncology" else "其他医学",
                    "combined_query": source_name,
                    "raw_answer_chunk": chunk,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "source_file": source_name,
                    "gatekeeper_label": label,
                    "upload_batch_id": upload_batch_id,
                    "is_uploaded": True,
                }
                page_content = f"【资料来源】：{source_name}\n【内容】：{chunk}"
                documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    @staticmethod
    def _sample_text(text_blocks: list[str]) -> str:
        sample_count = min(len(text_blocks), random.randint(3, 5))
        samples = random.sample(text_blocks, k=sample_count)
        return "\n\n".join(samples)[:4000]

    @classmethod
    def _extract_label(cls, raw_output: str) -> str:
        for label in cls.VALID_LABELS:
            if label in raw_output:
                return label
        raise ValueError(f"大模型数据分类结果无效：{raw_output!r}")

    @classmethod
    def _clean_text(cls, value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        text = MedicalDataProcessor.HTML_TAG_PATTERN.sub(" ", text)
        text = MedicalDataProcessor.WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()
