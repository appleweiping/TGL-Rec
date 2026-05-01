"""Robust parser for strict JSON LLM diagnostic outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from llm4rec.metrics.ranking import deduplicate_preserve_order


class LLMParseError(ValueError):
    """Raised when an LLM response cannot be parsed as recoverable JSON."""


@dataclass(frozen=True)
class ParsedLLMResponse:
    ranked_item_ids: list[str]
    invalid_item_ids: list[str]
    duplicate_item_ids: list[str]
    reasoning_summary: str
    evidence_used: list[dict[str, Any]]
    raw_output: str
    parse_success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_llm_response(raw_output: str, *, candidate_items: list[str]) -> ParsedLLMResponse:
    """Parse a recoverable JSON response and separate hallucinated item IDs."""

    payload_text = _extract_json_object(raw_output)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"Could not parse LLM JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMParseError("LLM output JSON must be an object.")
    ranked = payload.get("ranked_item_ids")
    if not isinstance(ranked, list) or not all(isinstance(item, str) for item in ranked):
        raise LLMParseError("ranked_item_ids must be list[str].")
    candidate_set = {str(item) for item in candidate_items}
    duplicates = _duplicates(ranked)
    deduped = deduplicate_preserve_order(ranked)
    valid = [item for item in deduped if item in candidate_set]
    invalid = [item for item in deduped if item not in candidate_set]
    evidence = payload.get("evidence_used", [])
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise LLMParseError("evidence_used must be a list when present.")
    normalized_evidence = [
        _normalize_evidence_item(item)
        for item in evidence
        if isinstance(item, dict)
    ]
    return ParsedLLMResponse(
        ranked_item_ids=valid,
        invalid_item_ids=invalid,
        duplicate_item_ids=duplicates,
        reasoning_summary=str(payload.get("reasoning_summary", "")),
        evidence_used=normalized_evidence,
        raw_output=raw_output,
        parse_success=True,
    )


def try_parse_llm_response(raw_output: str, *, candidate_items: list[str]) -> ParsedLLMResponse:
    """Parse an LLM response without crashing diagnostic runners."""

    try:
        return parse_llm_response(raw_output, candidate_items=candidate_items)
    except LLMParseError as exc:
        return ParsedLLMResponse(
            ranked_item_ids=[],
            invalid_item_ids=[],
            duplicate_item_ids=[],
            reasoning_summary="",
            evidence_used=[],
            raw_output=raw_output,
            parse_success=False,
            metadata={"parse_error": str(exc)},
        )


def _extract_json_object(text: str) -> str:
    stripped = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    if start < 0:
        raise LLMParseError("No JSON object found in LLM output.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    raise LLMParseError("Unclosed JSON object in LLM output.")


def _duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates


def _normalize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_item": None if item.get("source_item") is None else str(item.get("source_item")),
        "target_item": None if item.get("target_item") is None else str(item.get("target_item")),
        "text": str(item.get("text", "")),
        "type": str(item.get("type", "semantic")),
    }

