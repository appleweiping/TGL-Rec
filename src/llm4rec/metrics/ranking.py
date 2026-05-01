"""Ranking metrics for prediction rows."""

from __future__ import annotations

import math
from typing import Any, Iterable


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    """Remove duplicate item ids while preserving first occurrence."""

    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        item_id = str(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        output.append(item_id)
    return output


def recall_at_k(predicted_items: list[str], target_item: str, k: int) -> float:
    """Single-target Recall@K."""

    _validate_k(k)
    ranked = deduplicate_preserve_order(predicted_items)
    return float(str(target_item) in ranked[:k])


def hit_rate_at_k(predicted_items: list[str], target_item: str, k: int) -> float:
    """Single-target HitRate@K."""

    return recall_at_k(predicted_items, target_item, k)


def ndcg_at_k(predicted_items: list[str], target_item: str, k: int) -> float:
    """Single-target NDCG@K."""

    _validate_k(k)
    ranked = deduplicate_preserve_order(predicted_items)
    for rank, item_id in enumerate(ranked[:k], start=1):
        if item_id == str(target_item):
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_at_k(predicted_items: list[str], target_item: str, k: int) -> float:
    """Single-target MRR@K."""

    _validate_k(k)
    ranked = deduplicate_preserve_order(predicted_items)
    for rank, item_id in enumerate(ranked[:k], start=1):
        if item_id == str(target_item):
            return 1.0 / rank
    return 0.0


def coverage(prediction_rows: list[dict[str, Any]], item_catalog: set[str]) -> float:
    """Catalog coverage from valid predicted item ids."""

    if not item_catalog:
        return 0.0
    predicted = {
        str(item_id)
        for row in prediction_rows
        for item_id in row.get("predicted_items", [])
        if str(item_id) in item_catalog
    }
    return len(predicted) / float(len(item_catalog))


def aggregate_ranking_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Aggregate ranking metrics over prediction rows."""

    if not prediction_rows:
        raise ValueError("Cannot evaluate empty prediction rows.")
    totals: dict[str, float] = {}
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    for k in cutoffs:
        totals[f"Recall@{k}"] = 0.0
        totals[f"HitRate@{k}"] = 0.0
        totals[f"NDCG@{k}"] = 0.0
        totals[f"MRR@{k}"] = 0.0
    for row in prediction_rows:
        predicted = [str(item) for item in row.get("predicted_items", [])]
        target = str(row["target_item"])
        for k in cutoffs:
            totals[f"Recall@{k}"] += recall_at_k(predicted, target, k)
            totals[f"HitRate@{k}"] += hit_rate_at_k(predicted, target, k)
            totals[f"NDCG@{k}"] += ndcg_at_k(predicted, target, k)
            totals[f"MRR@{k}"] += mrr_at_k(predicted, target, k)
    denominator = float(len(prediction_rows))
    return {name: value / denominator for name, value in sorted(totals.items())}


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
