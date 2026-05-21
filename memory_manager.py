from __future__ import annotations

import json
import logging
from typing import Any

import ollama


LOGGER = logging.getLogger(__name__)


SUMMARY_PROMPT = (
    "你是一个专业的医疗档案员。请根据以下历史对话，提取并更新患者的核心病历档案。"
    "要求：仅保留提及的核心病种、当前具体症状以及已尝试的治疗手段，字数控制在 100 字以内。"
    "绝对禁止包含废话。"
)


def compress_memory(
    chat_history: list[dict[str, str]],
    clinical_summary: str = "",
    model: str = "qwen3:8b",
    trigger_messages: int = 8,
) -> tuple[list[dict[str, str]], str]:
    """Compress chat history into a short clinical summary when history is too long."""
    if len(chat_history) <= trigger_messages:
        return chat_history, clinical_summary

    latest_round = chat_history[-2:] if len(chat_history) >= 2 else chat_history[-1:]
    history_text = json.dumps(chat_history, ensure_ascii=False, indent=2)
    user_prompt = (
        f"已有患者摘要：{clinical_summary or '无'}\n\n"
        f"需要压缩的历史对话：\n{history_text}\n\n"
        "请输出更新后的患者核心病历档案。"
    )

    LOGGER.info("Compressing chat memory with %d messages.", len(chat_history))
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    updated_summary = response["message"]["content"].strip()
    LOGGER.info("Updated clinical summary: %s", updated_summary)
    return latest_round, updated_summary


def append_message(chat_history: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    """Append one normalized chat message."""
    normalized_role = "assistant" if role == "assistant" else "user"
    return [*chat_history, {"role": normalized_role, "content": content}]


def normalize_chat_history(value: Any) -> list[dict[str, str]]:
    """Normalize frontend chat history into a safe list of role/content messages."""
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized
