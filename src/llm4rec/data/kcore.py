"""K-core filtering for recommendation interactions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from llm4rec.data.filtering import item_key, user_key


def iterative_k_core(
    interactions: list[dict[str, Any]],
    *,
    user_k: int = 3,
    item_k: int | None = None,
    max_iterations: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Repeatedly apply user/item min-count constraints until stable."""

    min_item = user_k if item_k is None else item_k
    rows = list(interactions)
    iterations: list[dict[str, Any]] = []
    for index in range(1, max_iterations + 1):
        user_counts = Counter(user_key(row) for row in rows)
        item_counts = Counter(item_key(row) for row in rows)
        retained_users = {key for key, count in user_counts.items() if count >= user_k}
        retained_items = {key for key, count in item_counts.items() if count >= min_item}
        next_rows = [
            row for row in rows if user_key(row) in retained_users and item_key(row) in retained_items
        ]
        iterations.append(
            {
                "input_interactions": len(rows),
                "input_items": len(item_counts),
                "input_users": len(user_counts),
                "iteration": index,
                "output_interactions": len(next_rows),
                "output_items": len(retained_items),
                "output_users": len(retained_users),
                "removed_interactions": len(rows) - len(next_rows),
            }
        )
        if len(next_rows) == len(rows):
            rows = next_rows
            break
        rows = next_rows
    final_user_counts = Counter(user_key(row) for row in rows)
    final_item_counts = Counter(item_key(row) for row in rows)
    report = {
        "converged": not iterations or iterations[-1]["removed_interactions"] == 0,
        "input_interactions": len(interactions),
        "input_items": len({item_key(row) for row in interactions}),
        "input_users": len({user_key(row) for row in interactions}),
        "item_min_interactions": min_item,
        "items_still_below_threshold": sum(1 for count in final_item_counts.values() if count < min_item),
        "iterations": iterations,
        "num_iterations": len(iterations),
        "output_interactions": len(rows),
        "output_items": len(final_item_counts),
        "output_users": len(final_user_counts),
        "removed_items": len({item_key(row) for row in interactions}) - len(final_item_counts),
        "removed_users": len({user_key(row) for row in interactions}) - len(final_user_counts),
        "user_min_interactions": user_k,
        "users_still_below_threshold": sum(1 for count in final_user_counts.values() if count < user_k),
    }
    return rows, report
