"""Need-state encoder: computes user temporal state from interaction history.

This module is part of the TGL-Rec reportable scoring path. It extracts
features that characterize whether a user is in a stable-preference state
or a transition state, which the learned need-gate uses to decide how much
to trust temporal evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NeedState:
    """User temporal need-state vector."""

    drift_magnitude: float
    transition_pressure: float
    temporal_gap: float
    category_entropy: float
    history_length: int

    def to_vector(self) -> list[float]:
        return [
            self.drift_magnitude,
            self.transition_pressure,
            self.temporal_gap,
            self.category_entropy,
            min(1.0, self.history_length / 50.0),
        ]

    def to_dict(self) -> dict[str, float]:
        return {
            "drift_magnitude": self.drift_magnitude,
            "transition_pressure": self.transition_pressure,
            "temporal_gap": self.temporal_gap,
            "category_entropy": self.category_entropy,
            "history_length_norm": min(1.0, self.history_length / 50.0),
        }


class NeedStateEncoder:
    """Extracts temporal need-state from user history and TDIG."""

    def __init__(
        self,
        *,
        recent_window: int = 5,
        tau_max: float = 30 * 86400,
        transition_index: dict[str, list[dict[str, Any]]] | None = None,
        item_categories: dict[str, str] | None = None,
    ) -> None:
        self.recent_window = recent_window
        self.tau_max = tau_max
        self.transition_index = transition_index or {}
        self.item_categories = item_categories or {}

    def encode(
        self,
        *,
        history: list[str],
        timestamps: list[float] | None = None,
        prediction_timestamp: float | None = None,
    ) -> NeedState:
        n = len(history)
        if n == 0:
            return NeedState(0.0, 0.0, 1.0, 0.0, 0)

        w = min(self.recent_window, n)
        recent = history[-w:]
        all_cats = [self.item_categories.get(item, "unknown") for item in history]
        recent_cats = all_cats[-w:]

        drift_magnitude = self._compute_drift(all_cats, recent_cats)
        transition_pressure = self._compute_transition_pressure(history[-1], recent_cats[-1])
        temporal_gap = self._compute_temporal_gap(timestamps, prediction_timestamp)
        category_entropy = self._compute_entropy(recent_cats)

        return NeedState(
            drift_magnitude=drift_magnitude,
            transition_pressure=transition_pressure,
            temporal_gap=temporal_gap,
            category_entropy=category_entropy,
            history_length=n,
        )

    def _compute_drift(self, all_cats: list[str], recent_cats: list[str]) -> float:
        if len(all_cats) < 2:
            return 0.0
        all_dist = Counter(all_cats)
        recent_dist = Counter(recent_cats)
        all_total = sum(all_dist.values())
        recent_total = sum(recent_dist.values())
        if all_total == 0 or recent_total == 0:
            return 0.0
        all_keys = set(all_dist) | set(recent_dist)
        divergence = 0.0
        for cat in all_keys:
            p = all_dist.get(cat, 0) / all_total
            q = recent_dist.get(cat, 0) / recent_total
            divergence += abs(p - q)
        return min(1.0, divergence / 2.0)

    def _compute_transition_pressure(self, last_item: str, last_category: str) -> float:
        out_edges = self.transition_index.get(last_item, [])
        if not out_edges:
            return 0.0
        total_weight = 0.0
        cross_category_weight = 0.0
        for edge in out_edges:
            w = float(edge.get("weight", 1.0))
            total_weight += w
            target = str(edge.get("target", ""))
            target_cat = self.item_categories.get(target, "unknown")
            if target_cat != last_category:
                cross_category_weight += w
        if total_weight == 0:
            return 0.0
        return cross_category_weight / total_weight

    def _compute_temporal_gap(
        self, timestamps: list[float] | None, prediction_timestamp: float | None
    ) -> float:
        if not timestamps or prediction_timestamp is None:
            return 0.5
        last_t = timestamps[-1]
        gap = max(0.0, prediction_timestamp - last_t)
        return min(1.0, gap / self.tau_max)

    def _compute_entropy(self, categories: list[str]) -> float:
        if not categories:
            return 0.0
        counts = Counter(categories)
        total = len(categories)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(max(len(counts), 2))
        return entropy / max_entropy if max_entropy > 0 else 0.0
