"""Segment helpers for later per-domain and per-temporal analyses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


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
