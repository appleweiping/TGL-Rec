"""Structured-output schema helpers for API micro diagnostics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


STRUCTURED_OUTPUT_SCHEMA_VERSION = "phase3b.api_micro.schema.v1"
STRUCTURED_OUTPUT_NAME = "llm4rec_api_micro_rerank"
EVIDENCE_TYPES = (
    "history",
    "time_gap",
    "time_bucket",
    "transition",
    "time_window",
    "semantic",
    "contrastive",
)


def api_micro_response_schema() -> dict[str, Any]:
    """Return the strict JSON schema expected from API micro diagnostics."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ranked_item_ids", "reasoning_summary", "evidence_used"],
        "properties": {
            "ranked_item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate item IDs ordered from most to least recommended.",
            },
            "reasoning_summary": {
                "type": "string",
                "description": "A short audit-friendly explanation, not a full chain of thought.",
            },
            "evidence_used": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "source_item", "target_item", "text"],
                    "properties": {
                        "type": {"type": "string", "enum": list(EVIDENCE_TYPES)},
                        "source_item": {"type": ["string", "null"]},
                        "target_item": {"type": ["string", "null"]},
                        "text": {"type": "string"},
                    },
                },
            },
        },
    }


def openai_response_format(*, enabled: bool, strict: bool = True) -> dict[str, Any] | None:
    """Build an OpenAI-compatible response_format payload when enabled."""

    if not enabled:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": STRUCTURED_OUTPUT_NAME,
            "strict": bool(strict),
            "schema": api_micro_response_schema(),
        },
    }


def structured_output_metadata(*, enabled: bool, strict: bool = True) -> dict[str, Any]:
    """Return non-secret request metadata for structured output and cache keys."""

    metadata: dict[str, Any] = {
        "structured_output_enabled": bool(enabled),
        "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION
        if enabled
        else "disabled",
    }
    if enabled:
        metadata["structured_output_strict"] = bool(strict)
        metadata["structured_output_schema"] = deepcopy(api_micro_response_schema())
    return metadata
