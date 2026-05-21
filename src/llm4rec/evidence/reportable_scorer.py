"""Reportable scoring pipeline for TGL-Rec.

Replaces the deterministic smoke scorer with a trained pipeline:
1. NeedStateEncoder computes user temporal state
2. Evidence features extracted per candidate
3. LearnedNeedGate predicts α (temporal vs semantic balance)
4. Final score combines gated temporal + semantic + recency - trap penalty
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm4rec.evidence.base import Evidence
from llm4rec.evidence.need_gate import GateConfig, LearnedNeedGate
from llm4rec.evidence.need_state import NeedState, NeedStateEncoder


@dataclass(frozen=True)
class ReportableCandidateScore:
    """Score decomposition for one candidate under the reportable pipeline."""

    item_id: str
    alpha: float = 0.5
    temporal_score: float = 0.0
    semantic_score: float = 0.0
    recency_score: float = 0.0
    trap_penalty: float = 0.0
    confidence: float = 0.0
    total_score: float = 0.0
    evidence_features: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "alpha": self.alpha,
            "temporal_score": self.temporal_score,
            "semantic_score": self.semantic_score,
            "recency_score": self.recency_score,
            "trap_penalty": self.trap_penalty,
            "confidence": self.confidence,
            "total_score": self.total_score,
        }


class ReportableScorer:
    """Trained scoring pipeline with learned need-gate."""

    def __init__(
        self,
        *,
        gate: LearnedNeedGate | None = None,
        need_state_encoder: NeedStateEncoder | None = None,
        trap_penalty_weight: float = 0.3,
        confidence_weight: float = 0.15,
        recency_weight: float = 0.2,
    ) -> None:
        self.gate = gate or LearnedNeedGate()
        self.need_state_encoder = need_state_encoder or NeedStateEncoder()
        self.trap_penalty_weight = trap_penalty_weight
        self.confidence_weight = confidence_weight
        self.recency_weight = recency_weight
        self._need_state_cache: NeedState | None = None

    def set_user_context(
        self,
        *,
        history: list[str],
        timestamps: list[float] | None = None,
        prediction_timestamp: float | None = None,
    ) -> NeedState:
        self._need_state_cache = self.need_state_encoder.encode(
            history=history,
            timestamps=timestamps,
            prediction_timestamp=prediction_timestamp,
        )
        return self._need_state_cache

    def score_candidates(
        self,
        evidence: list[Evidence],
        candidate_items: list[str],
        *,
        history: list[str] | None = None,
        timestamps: list[float] | None = None,
        prediction_timestamp: float | None = None,
    ) -> dict[str, ReportableCandidateScore]:
        if self._need_state_cache is None:
            if history is None:
                raise ValueError("Must call set_user_context or pass history")
            self.set_user_context(
                history=history,
                timestamps=timestamps,
                prediction_timestamp=prediction_timestamp,
            )

        need_vec = self._need_state_cache.to_vector()
        output: dict[str, ReportableCandidateScore] = {}

        for item_id in candidate_items:
            item = str(item_id)
            rows = [r for r in evidence if str(r.target_item) == item]
            ef = self._extract_evidence_features(rows)
            alpha = self.gate.predict(need_vec, ef)
            temporal = ef[0] + ef[1] + ef[2] + ef[3] + ef[4] + ef[5]
            semantic = ef[6]
            recency = ef[8]
            trap = max(0.0, semantic - temporal) * self.trap_penalty_weight
            confidence = ef[9]
            total = (
                alpha * temporal
                + (1.0 - alpha) * semantic
                + self.recency_weight * recency
                + self.confidence_weight * confidence
                - trap
            )
            output[item] = ReportableCandidateScore(
                item_id=item,
                alpha=alpha,
                temporal_score=temporal,
                semantic_score=semantic,
                recency_score=recency,
                trap_penalty=trap,
                confidence=confidence,
                total_score=total,
                evidence_features=ef,
            )

        self._need_state_cache = None
        return output

    def prepare_gate_training_data(
        self,
        *,
        train_interactions: list[dict[str, Any]],
        evidence_by_user: dict[str, list[Evidence]],
        candidate_sets: dict[str, list[str]],
        ground_truth: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Generate training examples for the need-gate from train data.

        Positive: (user, ground_truth_item) where temporal evidence exists
        Negative: (user, high-semantic-sim item) that is NOT ground truth
        """
        examples: list[dict[str, Any]] = []

        for user_id, evidence_list in evidence_by_user.items():
            if user_id not in ground_truth or user_id not in candidate_sets:
                continue
            gt_item = ground_truth[user_id]
            candidates = candidate_sets[user_id]

            gt_rows = [r for r in evidence_list if str(r.target_item) == gt_item]
            gt_features = self._extract_evidence_features(gt_rows)
            gt_temporal = sum(gt_features[:6])

            if gt_temporal > 0:
                need_vec = self._need_state_cache.to_vector() if self._need_state_cache else [0.0] * 5
                examples.append({
                    "need_state": need_vec,
                    "evidence_features": gt_features,
                    "label": 1,
                    "weight": min(1.0, gt_temporal),
                    "user_id": user_id,
                    "item_id": gt_item,
                })

            for cand in candidates:
                if cand == gt_item:
                    continue
                cand_rows = [r for r in evidence_list if str(r.target_item) == cand]
                cand_features = self._extract_evidence_features(cand_rows)
                cand_semantic = cand_features[6] if len(cand_features) > 6 else 0.0
                cand_temporal = sum(cand_features[:6])
                if cand_semantic > 0.3 and cand_temporal < 0.1:
                    need_vec = self._need_state_cache.to_vector() if self._need_state_cache else [0.0] * 5
                    examples.append({
                        "need_state": need_vec,
                        "evidence_features": cand_features,
                        "label": 0,
                        "weight": cand_semantic,
                        "user_id": user_id,
                        "item_id": cand,
                    })

        return examples

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.gate.save(output_dir / "need_gate_weights.json")

    def load(self, output_dir: Path) -> None:
        gate_path = output_dir / "need_gate_weights.json"
        if gate_path.exists():
            self.gate.load(gate_path)

    @staticmethod
    def _extract_evidence_features(rows: list[Evidence]) -> list[float]:
        if not rows:
            return [0.0] * 10

        trans = 0.0
        prob = 0.0
        pmi = 0.0
        lift = 0.0
        da = 0.0
        win = 0.0
        sem = 0.0
        trap = 0.0
        rec = 0.0
        types: set[str] = set()
        support: set[str] = set()
        total_count = 0.0

        for row in rows:
            stats = row.stats
            types.add(row.evidence_type)
            support.update(str(s) for s in row.support_items)
            total_count += float(stats.get("transition_count") or stats.get("user_count") or 0.0)

            trans += math.log1p(float(stats.get("transition_count") or 0.0))
            prob = max(prob, float(stats.get("transition_probability") or 0.0))
            pmi = max(pmi, float(stats.get("pmi") or 0.0))
            lift = max(lift, math.log1p(max(0.0, float(stats.get("lift") or 0.0))))
            da += max(0.0, float(stats.get("direction_asymmetry") or 0.0))
            win += float(stats.get("time_window_score") or 0.0)
            sem = max(sem, float(stats.get("semantic_similarity") or 0.0))
            rec = max(rec, float(stats.get("recent_signal") or 0.0))

        temporal_total = trans + prob + pmi + lift + da + win
        trap = max(0.0, sem - temporal_total)

        diversity = min(1.0, (len(types) + len(support)) / 8.0)
        volume = 1.0 - math.exp(-total_count / 10.0)
        confidence = 0.5 * diversity + 0.5 * volume

        return [trans, prob, pmi, lift, da, win, sem, trap, rec, confidence]
