"""Sequence perturbation helpers for diagnostics."""

from __future__ import annotations

import random
from collections.abc import Sequence


def original_sequence(history: Sequence[str]) -> list[str]:
    """Return history as-is."""

    return [str(item) for item in history]


def reversed_sequence(history: Sequence[str]) -> list[str]:
    """Return history in reverse order."""

    return list(reversed(original_sequence(history)))


def shuffled_sequence(history: Sequence[str], *, seed: int) -> list[str]:
    """Return a deterministic seeded shuffle of history."""

    output = original_sequence(history)
    rng = random.Random(int(seed))
    rng.shuffle(output)
    return output


def recent_k_sequence(history: Sequence[str], *, k: int) -> list[str]:
    """Return the most recent k items."""

    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return []
    return original_sequence(history)[-k:]


def remove_recent_k_sequence(history: Sequence[str], *, k: int) -> list[str]:
    """Drop the most recent k items."""

    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return original_sequence(history)
    return original_sequence(history)[:-k]


def popularity_sorted_sequence(history: Sequence[str], popularity: dict[str, int | float]) -> list[str]:
    """Sort history by train popularity descending, then item id."""

    return sorted(
        original_sequence(history),
        key=lambda item_id: (-float(popularity.get(str(item_id), 0.0)), str(item_id)),
    )


def apply_sequence_transform(
    history: Sequence[str],
    *,
    transform: str,
    seed: int = 0,
    k: int = 3,
    popularity: dict[str, int | float] | None = None,
) -> list[str]:
    """Dispatch configured sequence transform."""

    if transform == "original":
        return original_sequence(history)
    if transform == "reversed":
        return reversed_sequence(history)
    if transform == "shuffled":
        return shuffled_sequence(history, seed=seed)
    if transform == "recent_k":
        return recent_k_sequence(history, k=k)
    if transform == "remove_recent_k":
        return remove_recent_k_sequence(history, k=k)
    if transform == "popularity_sorted":
        return popularity_sorted_sequence(history, popularity or {})
    raise ValueError(f"Unknown sequence transform: {transform}")
