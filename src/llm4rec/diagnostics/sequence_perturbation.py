"""Sequence perturbation artifact construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from llm4rec.data.sequence_transforms import apply_sequence_transform


DEFAULT_TRANSFORMS = [
    "original",
    "reversed",
    "shuffled",
    "recent_k",
    "remove_recent_k",
    "popularity_sorted",
]


def build_sequence_perturbation_artifact(
    interactions: list[dict[str, Any]],
    *,
    seed: int,
    recent_k: int,
    transforms: list[str] | None = None,
) -> dict[str, Any]:
    """Build per-user perturbed histories without scoring a model."""

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)
    popularity = Counter(str(row["item_id"]) for row in interactions if row.get("split") == "train")
    selected = transforms or DEFAULT_TRANSFORMS
    users: list[dict[str, Any]] = []
    for user_id in sorted(by_user):
        rows = sorted(
            by_user[user_id],
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            ),
        )
        history = [str(row["item_id"]) for row in rows]
        users.append(
            {
                "transforms": {
                    name: apply_sequence_transform(
                        history,
                        transform=name,
                        seed=seed,
                        k=recent_k,
                        popularity=popularity,
                    )
                    for name in selected
                },
                "user_id": user_id,
            }
        )
    return {
        "recent_k": int(recent_k),
        "seed": int(seed),
        "transforms": selected,
        "users": users,
    }
