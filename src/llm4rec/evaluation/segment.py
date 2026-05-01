"""Segment helpers for later per-domain and per-temporal analyses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from llm4rec.metrics.ranking import aggregate_ranking_metrics


def group_rows(
    rows: list[dict[str, Any]],
    *,
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, list[dict[str, Any]]]:
    """Group rows by a deterministic string key."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return dict(sorted(groups.items()))


def history_length_bucket(length: int) -> str:
    """Bucket user history lengths for segment metrics."""

    if length <= 0:
        return "empty"
    if length <= 2:
        return "1_2"
    if length <= 5:
        return "3_5"
    if length <= 10:
        return "6_10"
    return "gt_10"


def domain_segment(row: dict[str, Any]) -> str:
    """Return a row domain segment."""

    return str(row.get("domain") or "unknown")


def user_sparsity_segment(row: dict[str, Any]) -> str:
    """Return a user-sparsity segment from metadata history length."""

    metadata = row.get("metadata", {})
    return history_length_bucket(int(metadata.get("history_length", len(metadata.get("history", [])) or 0)))


def item_popularity_segment(item_id: str, popularity: dict[str, int]) -> str:
    """Return item-popularity segment."""

    count = int(popularity.get(str(item_id), 0))
    if count <= 0:
        return "unseen"
    if count <= 1:
        return "tail"
    if count <= 5:
        return "mid"
    return "head"


def evaluate_segments(
    prediction_rows: list[dict[str, Any]],
    *,
    ks: tuple[int, ...],
    segment_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """Compute ranking metrics per segment."""

    rows: list[dict[str, Any]] = []
    for segment, group in group_rows(prediction_rows, key_fn=segment_fn).items():
        metrics = aggregate_ranking_metrics(group, ks=ks)
        rows.append({"num_predictions": len(group), "segment": segment, **metrics})
    return rows
