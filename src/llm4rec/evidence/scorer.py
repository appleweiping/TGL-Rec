"""Deterministic Phase 5 evidence scoring."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm4rec.evidence.base import Evidence


DEFAULT_EVIDENCE_WEIGHTS = {
    "transition_weight": 1.0,
    "window_weight": 1.0,
    "semantic_weight": 1.0,
    "recency_weight": 1.0,
}


def score_evidence_for_candidate(
    evidence: list[Evidence],
    candidate_id: str,
    weights: dict[str, Any] | None = None,
) -> float:
    """Score one candidate from retrieved evidence."""

    merged = {**DEFAULT_EVIDENCE_WEIGHTS, **dict(weights or {})}
    score = 0.0
    for row in evidence:
        if str(row.target_item) != str(candidate_id):
            continue
        stats = row.stats
        score += float(merged["transition_weight"]) * float(stats.get("transition_count") or 0.0)
        score += float(merged["window_weight"]) * float(stats.get("time_window_score") or 0.0)
        score += float(merged["semantic_weight"]) * float(stats.get("semantic_similarity") or 0.0)
        score += float(merged["recency_weight"]) * float(stats.get("recent_signal") or 0.0)
    return float(score)


def score_candidates(
    evidence: list[Evidence],
    candidate_items: list[str],
    weights: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Score all candidates."""

    scores: dict[str, float] = defaultdict(float)
    for item_id in candidate_items:
        scores[str(item_id)] = score_evidence_for_candidate(evidence, str(item_id), weights)
    return dict(scores)
