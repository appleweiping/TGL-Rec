"""Strict JSON parsing for API LLM reranking outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class LLMJSONParseError(ValueError):
    """Raised when an LLM response cannot be parsed as the required JSON schema."""


@dataclass(frozen=True)
class ParsedRerankOutput:
    """Parsed reranking output with validity diagnostics."""

    ranked_item_ids: list[str]
    invalid_item_ids: list[str] = field(default_factory=list)
    duplicate_item_ids: list[str] = field(default_factory=list)
    evidence_usage: dict[str, bool] = field(default_factory=dict)
    raw_json: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_adherent(self) -> bool:
        return not self.invalid_item_ids


def parse_rerank_json(raw_output: str, *, candidate_item_ids: list[str]) -> ParsedRerankOutput:
    """Parse a strict JSON rerank response and validate candidate membership."""

    payload = _load_json_object(raw_output)
    ranked = payload.get("ranked_item_ids")
    if not isinstance(ranked, list) or not all(isinstance(item, str) for item in ranked):
        raise LLMJSONParseError("response must contain ranked_item_ids as a list of strings")
    candidate_set = {str(item) for item in candidate_item_ids}
    seen: set[str] = set()
    deduped: list[str] = []
    duplicates: list[str] = []
    invalid: list[str] = []
    for item in ranked:
        item_id = str(item)
        if item_id in seen:
            duplicates.append(item_id)
            continue
        seen.add(item_id)
        if item_id not in candidate_set:
            invalid.append(item_id)
            continue
        deduped.append(item_id)
    usage = payload.get("evidence_usage", {})
    if not isinstance(usage, dict):
        usage = {}
    normalized_usage = {
        "transition": bool(usage.get("transition", False)),
        "time": bool(usage.get("time", False)),
        "semantic": bool(usage.get("semantic", False)),
        "contrastive": bool(usage.get("contrastive", False)),
    }
    return ParsedRerankOutput(
        ranked_item_ids=deduped,
        invalid_item_ids=invalid,
        duplicate_item_ids=duplicates,
        evidence_usage=normalized_usage,
        raw_json=payload,
    )


def _load_json_object(raw_output: str) -> dict[str, Any]:
    text = str(raw_output).strip()
    if not text:
        raise LLMJSONParseError("empty LLM response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMJSONParseError("response is not valid JSON") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMJSONParseError(f"response JSON extraction failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMJSONParseError("response JSON must be an object")
    return parsed
