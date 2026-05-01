"""Edge-weight helpers for temporal graphs."""

from __future__ import annotations

import math


def exponential_decay_weight(
    gap_seconds: int | float,
    *,
    half_life_seconds: int | float,
) -> float:
    """Return an exponential time-decay weight with configurable half-life."""

    gap = float(gap_seconds)
    half_life = float(half_life_seconds)
    if gap < 0:
        raise ValueError(f"gap_seconds must be non-negative, got {gap_seconds}")
    if half_life <= 0:
        raise ValueError(f"half_life_seconds must be positive, got {half_life_seconds}")
    return math.pow(0.5, gap / half_life)
