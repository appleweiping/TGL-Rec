"""Novelty metrics from item popularity."""

from __future__ import annotations

import math
from typing import Any


def item_novelty(item_id: str, popularity: dict[str, int], *, total_interactions: int | None = None) -> float:
    """Self-information novelty using smoothed popularity."""

    total = int(total_interactions or sum(popularity.values()) or 1)
    count = int(popularity.get(str(item_id), 0))
    probability = (count + 1.0) / float(total + len(popularity) + 1)
    return -math.log2(probability)


def aggregate_novelty(
    prediction_rows: list[dict[str, Any]],
    popularity: dict[str, int],
    *,
    k: int | None = None,
) -> float:
    """Mean novelty over predicted items."""

    values: list[float] = []
    total = sum(popularity.values())
    for row in prediction_rows:
        items = [str(item) for item in row.get("predicted_items", [])]
        values.extend(item_novelty(item, popularity, total_interactions=total) for item in (items[:k] if k else items))
    return sum(values) / float(len(values) or 1)
