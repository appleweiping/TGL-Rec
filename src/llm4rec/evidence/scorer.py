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
    "need_state_weight": 0.35,
    "evidence_confidence_weight": 0.15,
    "drift_alignment_weight": 0.2,
}


@dataclass(frozen=True)
class CandidateFactorScore:
    """Factorized score for one candidate."""

    item_id: str
    contrastive_score: float = 0.0
    drift_score: float = 0.0
    evidence_counts: dict[str, int] = field(default_factory=dict)
    gate: float = 0.5
    evidence_confidence: float = 0.0
    need_state_score: float = 0.0
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
            "evidence_confidence": self.evidence_confidence,
            "gate": self.gate,
            "item_id": self.item_id,
            "need_state_score": self.need_state_score,
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
        need_state = 0.0
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row.evidence_type] += 1
            stats = row.stats
            metadata = row.metadata
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
                drift_signal = float(stats.get("recent_signal") or 0.0)
                drift += float(merged["drift_weight"]) * drift_signal
                need_state += float(merged["drift_alignment_weight"]) * _drift_alignment(metadata, drift_signal)
            need_state += _need_state_contribution(row.evidence_type, stats, metadata, merged)
        confidence = _evidence_confidence(rows)
        gate = _need_gate(
            temporal=temporal + contrastive + drift + need_state,
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
            gate * (temporal + contrastive + drift + need_state)
            + (1.0 - gate) * semantic
            + recency
            + float(merged["evidence_confidence_weight"]) * confidence
            - semantic_trap_penalty
        )
        output[item] = CandidateFactorScore(
            item_id=item,
            contrastive_score=float(contrastive),
            drift_score=float(drift),
            evidence_counts=dict(sorted(counts.items())),
            evidence_confidence=float(confidence),
            gate=float(gate),
            need_state_score=float(need_state),
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


def _need_state_contribution(
    evidence_type: str,
    stats: dict[str, Any],
    metadata: dict[str, Any],
    weights: dict[str, Any],
) -> float:
    """Score whether a candidate matches the user's current temporal need state."""

    weight = float(weights["need_state_weight"])
    if weight <= 0.0:
        return 0.0
    recent = float(stats.get("recent_signal") or 0.0)
    transition = float(stats.get("transition_probability") or 0.0)
    same_recent_category = _same_value(metadata.get("target_category"), metadata.get("recent_category"))
    same_source_category = _same_value(metadata.get("target_category"), metadata.get("source_category"))
    if evidence_type == "history":
        return weight * recent * (1.0 if same_recent_category or same_source_category else 0.25)
    if evidence_type in {"transition", "time_window", "contrastive"}:
        category_alignment = 0.5 if same_recent_category or same_source_category else 1.0
        return weight * max(recent, transition) * category_alignment
    if evidence_type == "semantic":
        return weight * 0.25 * float(stats.get("semantic_similarity") or 0.0)
    return 0.0


def _drift_alignment(metadata: dict[str, Any], drift_signal: float) -> float:
    """Reward candidates that match recent drift destination over stale profile."""

    if drift_signal <= 0.0:
        return 0.0
    target = metadata.get("target_category")
    drift_to = metadata.get("drift_to")
    drift_from = metadata.get("drift_from")
    if _same_value(target, drift_to) and not _same_value(target, drift_from):
        return float(drift_signal)
    return 0.0


def _evidence_confidence(rows: list[Evidence]) -> float:
    """Confidence from diverse train-only evidence support, bounded to [0, 1]."""

    if not rows:
        return 0.0
    support = set()
    types = set()
    total_count = 0.0
    for row in rows:
        types.add(row.evidence_type)
        support.update(str(item) for item in row.support_items)
        total_count += float(row.stats.get("transition_count") or row.stats.get("user_count") or 0.0)
    diversity = min(1.0, (len(types) + len(support)) / 8.0)
    volume = 1.0 - math.exp(-total_count / 10.0)
    return max(0.0, min(1.0, 0.5 * diversity + 0.5 * volume))


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return str(left) == str(right)
