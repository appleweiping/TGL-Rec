"""Deterministic mock LLM provider for smoke-only diagnostics."""

from __future__ import annotations

import json
import time
from typing import Any

from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.safety import ensure_mock_allowed


class MockLLMProvider:
    """A deterministic test provider, explicitly forbidden for reportable runs."""

    provider_name = "mock"

    def __init__(self, *, mode: str = "identity", model: str = "mock-llm", run_mode: str = "diagnostic_mock"):
        self.mode = str(mode)
        self.model = str(model)
        self.run_mode = str(run_mode)
        ensure_mock_allowed(self.run_mode)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a deterministic JSON response for one mode."""

        ensure_mock_allowed(str(request.metadata.get("run_mode", self.run_mode)))
        start = time.perf_counter()
        candidates = [str(item) for item in request.candidate_item_ids]
        ranked, evidence = self._rank(candidates, request.metadata)
        payload = {
            "evidence_used": evidence,
            "ranked_item_ids": ranked,
            "reasoning_summary": f"mock {self.mode} diagnostic output",
        }
        raw_output = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_tokens = max(1, len(request.prompt.split()))
        completion_tokens = max(1, len(raw_output.split()))
        return LLMResponse(
            raw_output=raw_output,
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            metadata={"mock_mode": self.mode},
        )

    def _rank(
        self,
        candidates: list[str],
        metadata: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, str | None]]]:
        if self.mode == "identity":
            return candidates, [_semantic_evidence(candidates)]
        if self.mode == "reverse":
            return list(reversed(candidates)), [_semantic_evidence(candidates)]
        if self.mode == "hallucinating":
            return ["not_in_candidates", *candidates], [
                {
                    "source_item": None,
                    "target_item": "not_in_candidates",
                    "text": "intentional invalid item for parser tests",
                    "type": "semantic",
                }
            ]
        if self.mode == "transition_aware":
            scores = {
                str(item): float(score)
                for item, score in dict(metadata.get("transition_scores", {})).items()
            }
            ordered = sorted(candidates, key=lambda item: (-scores.get(item, 0.0), item))
            return ordered, _evidence_from_rows(metadata.get("transition_evidence", []), "transition")
        if self.mode == "time_bucket_aware":
            scores = {
                str(item): float(score)
                for item, score in dict(metadata.get("time_window_scores", {})).items()
            }
            if not scores:
                scores = {
                    str(item): float(score)
                    for item, score in dict(metadata.get("time_bucket_scores", {})).items()
                }
            ordered = sorted(candidates, key=lambda item: (-scores.get(item, 0.0), item))
            return ordered, _evidence_from_rows(metadata.get("time_window_evidence", []), "time_window")
        raise ValueError(f"Unknown MockLLMProvider mode: {self.mode}")


def _semantic_evidence(candidates: list[str]) -> dict[str, str | None]:
    target = candidates[0] if candidates else None
    return {
        "source_item": None,
        "target_item": target,
        "text": "mock semantic candidate-order evidence",
        "type": "semantic",
    }


def _evidence_from_rows(rows: Any, evidence_type: str) -> list[dict[str, str | None]]:
    output: list[dict[str, str | None]] = []
    if not isinstance(rows, list):
        return output
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        output.append(
            {
                "source_item": None if row.get("source_item") is None else str(row.get("source_item")),
                "target_item": None if row.get("target_item") is None else str(row.get("target_item")),
                "text": (
                    f"{evidence_type} count={row.get('transition_count', row.get('count', ''))} "
                    f"bucket={row.get('dominant_gap_bucket', '')}"
                ).strip(),
                "type": evidence_type,
            }
        )
    return output or [_semantic_evidence([])]

