"""Parsing, hallucination, and grounding audit for Phase 3B API outputs."""

from __future__ import annotations

from typing import Any

from llm4rec.diagnostics.llm_grounding import evaluate_evidence_grounding
from llm4rec.llm.base import LLMResponse
from llm4rec.prompts.parsers import try_parse_llm_response


def audit_api_micro_response(
    *,
    sample: dict[str, Any],
    prompt_variant: str,
    prompt_version: str,
    response: LLMResponse,
    transition_edges: dict[tuple[str, str], dict[str, Any]],
    time_window_edges: dict[tuple[str, str], dict[str, Any]],
    time_bucket_by_pair: dict[tuple[str, str], str],
    run_mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Parse one API response into a prediction row plus optional failure audits."""

    parsed = try_parse_llm_response(response.raw_output, candidate_items=sample["candidate_items"])
    grounding = evaluate_evidence_grounding(
        parsed.evidence_used,
        history_items=sample["history"],
        candidate_items=sample["candidate_items"],
        transition_edges=transition_edges,
        time_window_edges=time_window_edges,
        time_bucket_by_pair=time_bucket_by_pair,
    )
    usage = response_usage(response)
    predicted_items = parsed.ranked_item_ids + parsed.invalid_item_ids
    prediction = {
        "candidate_items": sample["candidate_items"],
        "domain": sample.get("domain"),
        "metadata": {
            "cache_hit": response.cache_hit,
            "case_group": sample.get("case_group", sample.get("group")),
            "duplicate_item_ids": parsed.duplicate_item_ids,
            "evidence_source_item": sample.get("evidence_source_item"),
            "evidence_target_item": sample.get("evidence_target_item"),
            "grounding": grounding,
            "invalid_item_ids": parsed.invalid_item_ids,
            "llm_usage": usage,
            "parse_error": parsed.metadata.get("parse_error"),
            "parse_success": parsed.parse_success,
            "phase": "phase3b_api_micro",
            "prompt_variant": prompt_variant,
            "prompt_version": prompt_version,
            "reasoning_summary": parsed.reasoning_summary,
            "run_mode": run_mode,
            "sample_group": sample.get("case_group", sample.get("group")),
            "sample_id": sample["sample_id"],
        },
        "method": "api_micro_llm_rerank",
        "predicted_items": predicted_items,
        "raw_output": response.raw_output,
        "scores": [float(len(predicted_items) - index) for index, _ in enumerate(predicted_items)],
        "target_item": sample["target_item"],
        "user_id": sample["user_id"],
    }
    parse_failure = None
    if not parsed.parse_success:
        parse_failure = {
            "parse_error": parsed.metadata.get("parse_error"),
            "prompt_variant": prompt_variant,
            "raw_output": response.raw_output,
            "sample_id": sample["sample_id"],
        }
    hallucination = None
    if parsed.invalid_item_ids:
        hallucination = {
            "invalid_item_ids": parsed.invalid_item_ids,
            "prompt_variant": prompt_variant,
            "raw_output": response.raw_output,
            "sample_id": sample["sample_id"],
        }
    return prediction, parse_failure, hallucination


def response_usage(response: LLMResponse) -> dict[str, Any]:
    """Return the usage subset stored on predictions and raw-output rows."""

    return {
        "cache_hit": response.cache_hit,
        "completion_tokens": response.completion_tokens,
        "latency_ms": response.latency_ms,
        "prompt_tokens": response.prompt_tokens,
        "total_tokens": response.total_tokens,
    }


def raw_output_row(
    *,
    sample_id: str,
    prompt_variant: str,
    response: LLMResponse,
) -> dict[str, Any]:
    """Build a raw-output audit row without secrets."""

    return {
        "cache_hit": response.cache_hit,
        "metadata": response.metadata,
        "model": response.model,
        "prompt_variant": prompt_variant,
        "provider": response.provider,
        "raw_output": response.raw_output,
        "sample_id": sample_id,
        "usage": response_usage(response),
    }
