"""Deterministic split helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def leave_one_out_split(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign train/valid/test labels per user using the final two events."""

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(dict(row))

    output: list[dict[str, Any]] = []
    for user_id in sorted(by_user):
        rows = sorted(
            by_user[user_id],
            key=lambda row: (
                float(row["timestamp"]) if row["timestamp"] is not None else -1.0,
                str(row["item_id"]),
            ),
        )
        if len(rows) < 3:
            raise ValueError(f"leave-one-out requires at least 3 interactions for user {user_id}")
        for index, row in enumerate(rows):
            labeled = dict(row)
            if index == len(rows) - 2:
                labeled["split"] = "valid"
            elif index == len(rows) - 1:
                labeled["split"] = "test"
            else:
                labeled["split"] = "train"
            output.append(labeled)
    return sorted(
        output,
        key=lambda row: (
            str(row["user_id"]),
            float(row["timestamp"]) if row["timestamp"] is not None else -1.0,
            str(row["item_id"]),
        ),
    )


def build_user_histories(labeled_interactions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build ordered full user histories from labeled interactions."""

    histories: dict[str, list[str]] = defaultdict(list)
    for row in sorted(
        labeled_interactions,
        key=lambda value: (
            str(value["user_id"]),
            float(value["timestamp"]) if value["timestamp"] is not None else -1.0,
            str(value["item_id"]),
        ),
    ):
        histories[str(row["user_id"])].append(str(row["item_id"]))
    return dict(histories)
