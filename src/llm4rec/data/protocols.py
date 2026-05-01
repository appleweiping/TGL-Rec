"""Dataset protocol helpers for Phase 2B diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def temporal_split(
    interactions: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    """Apply a simple global temporal split."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in [0, 1), got {valid_ratio}")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")
    rows = sorted(
        [dict(row) for row in interactions],
        key=lambda row: (
            float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
            str(row["user_id"]),
            str(row["item_id"]),
        ),
    )
    n_rows = len(rows)
    train_cut = int(n_rows * train_ratio)
    valid_cut = int(n_rows * (train_ratio + valid_ratio))
    for index, row in enumerate(rows):
        if index < train_cut:
            row["split"] = "train"
        elif index < valid_cut:
            row["split"] = "valid"
        else:
            row["split"] = "test"
    return rows


def filter_min_interactions(
    interactions: list[dict[str, Any]],
    *,
    min_user_interactions: int,
) -> list[dict[str, Any]]:
    """Keep only users with enough interactions for leave-one-out diagnostics."""

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)
    output: list[dict[str, Any]] = []
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        if len(rows) >= min_user_interactions:
            output.extend(rows)
    return output


def subsample_interactions(
    interactions: list[dict[str, Any]],
    *,
    max_users: int | None = None,
    max_items: int | None = None,
    max_interactions: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically subsample users/items/interactions by sorted ids and time."""

    rows = list(interactions)
    if max_users is not None and max_users > 0:
        users = sorted({str(row["user_id"]) for row in rows})[: int(max_users)]
        rows = [row for row in rows if str(row["user_id"]) in set(users)]
    if max_items is not None and max_items > 0:
        items = sorted({str(row["item_id"]) for row in rows})[: int(max_items)]
        rows = [row for row in rows if str(row["item_id"]) in set(items)]
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["user_id"]),
            float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
            str(row["item_id"]),
        ),
    )
    if max_interactions is not None and max_interactions > 0:
        rows = rows[: int(max_interactions)]
    return rows
