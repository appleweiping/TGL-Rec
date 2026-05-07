"""Auditable temporal-need evidence scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from llm4rec.evidence.base import Evidence


DEFAULT_EVIDENCE_WEIGHTS = {
    "transition_weight": 1.0,
    "window_weight": 1.0,
    "semantic_weight": 1.0,
    "recency_weight": 1.0,
    "drift_weight": 0.5,
    "contrastive_transition_weight": 0.5,
    "semantic_trap_penalty": 0.0,
    "transition_probability_weight": 1.0,
    "pmi_weight": 0.5,
    "lift_weight": 0.1,
    "direction_asymmetry_weight": 0.25,
    "gate_bias": 0.0,
    "gate_temperature": 1.0,
}


@dataclass(frozen=True)
class CandidateFactorScore:
    """Factorized score for one candidate."""

    item_id: str
    contrastive_score: float = 0.0
    drift_score: float = 0.0
    evidence_counts: dict[str, int] = field(default_factory=dict)
    gate: float = 0.5
    recency_score: float = 0.0
    semantic_score: float = 0.0
    semantic_trap_penalty: float = 0.0
    temporal_score: float = 0.0
    total_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable score decomposition."""

        return {
            "contrastive_score": self.contrastive_score,
            "drift_score": self.drift_score,
            "evidence_counts": dict(self.evidence_counts),
            "gate": self.gate,
            "item_id": self.item_id,
            "recency_score": self.recency_score,
            "semantic_score": self.semantic_score,
            "semantic_trap_penalty": self.semantic_trap_penalty,
            "temporal_score": self.temporal_score,
            "total_score": self.total_score,
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
    """Score all candidates with the legacy additive scorer."""

    scores: dict[str, float] = defaultdict(float)
    for item_id in candidate_items:
        scores[str(item_id)] = score_evidence_for_candidate(evidence, str(item_id), weights)
    return dict(scores)


def factorized_score_candidates(
    evidence: list[Evidence],
    candidate_items: list[str],
    weights: dict[str, Any] | None = None,
) -> dict[str, CandidateFactorScore]:
    """Score candidates with an explicit semantic-vs-temporal need gate.

    The gate is the main TGL-Rec scoring primitive: it increases reliance on
    temporal transition evidence when transition/window/contrastive support is
    stronger than semantic-only similarity, and falls back toward semantic
    evidence when temporal support is absent. This keeps the method auditable and
    ablatable instead of hiding behavior in a single prompt.
    """

    merged = {**DEFAULT_EVIDENCE_WEIGHTS, **dict(weights or {})}
    output: dict[str, CandidateFactorScore] = {}
    for item_id in candidate_items:
        item = str(item_id)
        rows = [row for row in evidence if str(row.target_item) == item]
        temporal = 0.0
        semantic = 0.0
        recency = 0.0
        contrastive = 0.0
        drift = 0.0
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row.evidence_type] += 1
            stats = row.stats
            transition = math.log1p(float(stats.get("transition_count") or 0.0))
            window = float(stats.get("time_window_score") or 0.0)
            temporal += float(merged["transition_weight"]) * transition
            temporal += float(merged["transition_probability_weight"]) * float(
                stats.get("transition_probability") or 0.0
            )
            temporal += float(merged["pmi_weight"]) * max(0.0, float(stats.get("pmi") or 0.0))
            temporal += float(merged["lift_weight"]) * math.log1p(max(0.0, float(stats.get("lift") or 0.0)))
            temporal += float(merged["direction_asymmetry_weight"]) * max(
                0.0,
                float(stats.get("direction_asymmetry") or 0.0),
            )
            temporal += float(merged["window_weight"]) * window
            semantic += float(merged["semantic_weight"]) * float(stats.get("semantic_similarity") or 0.0)
            recency += float(merged["recency_weight"]) * float(stats.get("recent_signal") or 0.0)
            if row.evidence_type == "contrastive":
                contrastive += float(merged["contrastive_transition_weight"]) * transition
            if row.evidence_type == "user_drift":
                drift += float(merged["drift_weight"]) * float(stats.get("recent_signal") or 0.0)
        gate = _need_gate(
            temporal=temporal + contrastive + drift,
            semantic=semantic,
            bias=float(merged["gate_bias"]),
            temperature=float(merged["gate_temperature"]),
        )
        semantic_trap_penalty = _semantic_trap_penalty(
            temporal=temporal + contrastive,
            semantic=semantic,
            weight=float(merged["semantic_trap_penalty"]),
        )
        total = (
            gate * (temporal + contrastive + drift)
            + (1.0 - gate) * semantic
            + recency
            - semantic_trap_penalty
        )
        output[item] = CandidateFactorScore(
            item_id=item,
            contrastive_score=float(contrastive),
            drift_score=float(drift),
            evidence_counts=dict(sorted(counts.items())),
            gate=float(gate),
            recency_score=float(recency),
            semantic_score=float(semantic),
            semantic_trap_penalty=float(semantic_trap_penalty),
            temporal_score=float(temporal),
            total_score=float(total),
        )
    return output


def _need_gate(*, temporal: float, semantic: float, bias: float, temperature: float) -> float:
    temperature = max(float(temperature), 1e-6)
    logit = (float(temporal) - float(semantic) + float(bias)) / temperature
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _semantic_trap_penalty(*, temporal: float, semantic: float, weight: float) -> float:
    if weight <= 0.0:
        return 0.0
    return max(0.0, float(semantic) - float(temporal)) * float(weight)
