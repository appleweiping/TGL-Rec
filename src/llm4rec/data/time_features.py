"""Time-gap and recency features for sequence diagnostics."""

from __future__ import annotations

from typing import Any

HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS
WEEK_SECONDS = 7 * DAY_SECONDS
MONTH_SECONDS = 30 * DAY_SECONDS


def bucket_time_gap(gap_seconds: int | float | None) -> str:
    """Bucket a non-negative time gap."""

    if gap_seconds is None:
        return "unknown"
    gap = float(gap_seconds)
    if gap < 0:
        raise ValueError(f"time gap must be non-negative, got {gap_seconds}")
    if gap <= HOUR_SECONDS:
        return "same_session"
    if gap <= DAY_SECONDS:
        return "same_day"
    if gap <= WEEK_SECONDS:
        return "same_week"
    if gap <= MONTH_SECONDS:
        return "same_month"
    return "old"


def consecutive_time_gaps(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute absolute timestamps and gaps for an ordered user sequence."""

    rows = sorted(
        interactions,
        key=lambda row: (
            float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
            str(row["item_id"]),
        ),
    )
    output: list[dict[str, Any]] = []
    previous_ts: float | None = None
    for index, row in enumerate(rows):
        timestamp = None if row.get("timestamp") is None else float(row["timestamp"])
        gap = None if previous_ts is None or timestamp is None else timestamp - previous_ts
        output.append(
            {
                "gap_bucket": bucket_time_gap(gap),
                "gap_seconds": gap,
                "index": index,
                "item_id": str(row["item_id"]),
                "timestamp": timestamp,
                "user_id": str(row["user_id"]),
            }
        )
        previous_ts = timestamp
    return output


def recency_blocks(history: list[str]) -> dict[str, list[str]]:
    """Split a history into short, mid, and long-term blocks."""

    items = [str(item) for item in history]
    n_items = len(items)
    if n_items == 0:
        return {"long_term": [], "mid_term": [], "short_term": []}
    first = n_items // 3
    second = (2 * n_items) // 3
    return {
        "long_term": items[:first],
        "mid_term": items[first:second],
        "short_term": items[second:],
    }


def build_time_feature_rows(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-interaction time features grouped by user."""

    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in interactions:
        by_user.setdefault(str(row["user_id"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for user_id in sorted(by_user):
        rows.extend(consecutive_time_gaps(by_user[user_id]))
    return rows
