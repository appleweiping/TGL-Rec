"""Ranking metrics for recommendation experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral


def rank_by_score(scores: Mapping[int | str, float]) -> list[int | str]:
    """Rank candidate items by score with deterministic item-id tie breaking."""

    return [
        item_id
        for item_id, _ in sorted(
            scores.items(),
            key=lambda pair: (-pair[1], _tie_break_key(pair[0])),
        )
    ]


def _tie_break_key(item_id: int | str) -> tuple[int, int | str]:
    if isinstance(item_id, Integral):
        return (0, int(item_id))
    return (1, str(item_id))


def hit_rate_at_k(ranked_items: Sequence[int | str], positive_item: int | str, k: int) -> float:
    """Return 1.0 if the positive item appears in the top-k ranking."""

    _validate_k(k)
    return float(positive_item in ranked_items[:k])


def ndcg_at_k(ranked_items: Sequence[int | str], positive_item: int | str, k: int) -> float:
    """NDCG@K for a single relevant item."""

    _validate_k(k)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == positive_item:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_at_k(ranked_items: Sequence[int | str], positive_item: int | str, k: int) -> float:
    """MRR@K for a single relevant item."""

    _validate_k(k)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == positive_item:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    rankings: Mapping[int | str, Sequence[int | str]],
    positives: Mapping[int | str, int | str],
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Aggregate HR, NDCG, and MRR over users.

    Args:
        rankings: User id to ranked item ids. Higher rank is earlier in sequence.
        positives: User id to one held-out relevant item.
        ks: Cutoffs to evaluate.
    """

    if not positives:
        raise ValueError("Cannot evaluate empty positives.")
    missing = sorted(set(positives) - set(rankings))
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Missing rankings for {len(missing)} users, e.g. {preview}")

    cutoffs = tuple(sorted(set(ks)))
    totals = {f"HR@{k}": 0.0 for k in cutoffs}
    totals.update({f"NDCG@{k}": 0.0 for k in cutoffs})
    totals.update({f"MRR@{k}": 0.0 for k in cutoffs})
    for user_id, positive in positives.items():
        ranked = rankings[user_id]
        for k in cutoffs:
            totals[f"HR@{k}"] += hit_rate_at_k(ranked, positive, k)
            totals[f"NDCG@{k}"] += ndcg_at_k(ranked, positive, k)
            totals[f"MRR@{k}"] += mrr_at_k(ranked, positive, k)
    denominator = float(len(positives))
    return {name: value / denominator for name, value in sorted(totals.items())}


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
