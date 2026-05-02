"""Shared filtering helpers for recommendation interaction tables."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def user_key(row: dict[str, Any]) -> str:
    """Return the normalized user key used by filtering."""

    return str(row.get("user_id"))


def item_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return a domain-aware item key."""

    return (str(row.get("domain") or ""), str(row.get("item_id")))


def filter_by_min_counts(
    interactions: Iterable[dict[str, Any]],
    *,
    user_min_interactions: int = 3,
    item_min_interactions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a single min-count filtering pass to in-memory interactions."""

    rows = list(interactions)
    user_counts = Counter(user_key(row) for row in rows)
    item_counts = Counter(item_key(row) for row in rows)
    min_item = user_min_interactions if item_min_interactions is None else item_min_interactions
    retained = [
        row
        for row in rows
        if user_counts[user_key(row)] >= user_min_interactions and item_counts[item_key(row)] >= min_item
    ]
    after_user_counts = Counter(user_key(row) for row in retained)
    after_item_counts = Counter(item_key(row) for row in retained)
    report = {
        "input_interactions": len(rows),
        "input_items": len(item_counts),
        "input_users": len(user_counts),
        "item_min_interactions": min_item,
        "items_still_below_threshold": sum(1 for count in after_item_counts.values() if count < min_item),
        "output_interactions": len(retained),
        "output_items": len(after_item_counts),
        "output_users": len(after_user_counts),
        "removed_items": len(item_counts) - len(after_item_counts),
        "removed_users": len(user_counts) - len(after_user_counts),
        "retained_interaction_ratio": _ratio(len(retained), len(rows)),
        "retained_item_ratio": _ratio(len(after_item_counts), len(item_counts)),
        "retained_user_ratio": _ratio(len(after_user_counts), len(user_counts)),
        "user_min_interactions": user_min_interactions,
        "users_still_below_threshold": sum(
            1 for count in after_user_counts.values() if count < user_min_interactions
        ),
    }
    return retained, report


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
