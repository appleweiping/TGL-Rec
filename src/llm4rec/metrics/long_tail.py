"""Long-tail and popularity-stratified metrics."""

from __future__ import annotations

from typing import Any


def popularity_counts(interactions: list[dict[str, Any]]) -> dict[str, int]:
    """Count item popularity from interactions."""

    counts: dict[str, int] = {}
    for row in interactions:
        item = str(row["item_id"])
        counts[item] = counts.get(item, 0) + 1
    return counts


def long_tail_items(counts: dict[str, int], *, quantile: float = 0.2) -> set[str]:
    """Return the least-popular item IDs by count quantile."""

    if not counts:
        return set()
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    cutoff = max(1, int(round(len(ordered) * float(quantile))))
    return {item for item, _count in ordered[:cutoff]}


def long_tail_ratio(prediction_rows: list[dict[str, Any]], long_tail: set[str], *, k: int | None = None) -> float:
    """Fraction of predicted items that are long-tail."""

    total = 0
    hits = 0
    tail = {str(item) for item in long_tail}
    for row in prediction_rows:
        items = [str(item) for item in row.get("predicted_items", [])]
        for item in items[:k] if k is not None else items:
            total += 1
            hits += 1 if item in tail else 0
    return hits / float(total or 1)


def popularity_bucket(item_id: str, counts: dict[str, int]) -> str:
    """Assign an item to a coarse popularity bucket."""

    value = int(counts.get(str(item_id), 0))
    if value == 0:
        return "unseen"
    if value <= 1:
        return "tail"
    if value <= 5:
        return "mid"
    return "head"


def popularity_bucket_metrics(
    prediction_rows: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    k: int | None = None,
) -> dict[str, float]:
    """Distribution of predicted items by popularity bucket."""

    bucket_counts = {"head": 0, "mid": 0, "tail": 0, "unseen": 0}
    total = 0
    for row in prediction_rows:
        items = [str(item) for item in row.get("predicted_items", [])]
        for item in items[:k] if k is not None else items:
            bucket_counts[popularity_bucket(item, counts)] += 1
            total += 1
    return {f"popularity_bucket_{name}_rate": count / float(total or 1) for name, count in bucket_counts.items()}
