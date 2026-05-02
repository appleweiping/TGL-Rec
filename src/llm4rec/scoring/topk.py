"""Deterministic top-k helpers for candidate score matrices."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Any


def top_k_items_and_scores(
    scores: Sequence[float],
    item_ids: Sequence[str],
    *,
    top_n: int,
) -> tuple[list[str], list[float]]:
    """Select top-N by score descending, then item_id ascending."""

    if len(scores) != len(item_ids):
        raise ValueError("scores and item_ids must have identical length")
    limit = min(int(top_n), len(item_ids))
    if limit <= 0:
        return [], []
    ranked = heapq.nsmallest(
        limit,
        ((-float(score), str(item_id), index) for index, (score, item_id) in enumerate(zip(scores, item_ids))),
    )
    indices = [index for _neg_score, _item_id, index in ranked]
    return [str(item_ids[index]) for index in indices], [float(scores[index]) for index in indices]


def top_k_from_score_matrix(
    score_matrix: Any,
    candidate_item_ids: Sequence[Sequence[str]],
    *,
    top_n: int,
) -> list[tuple[list[str], list[float]]]:
    """Apply deterministic top-N selection row-wise."""

    output: list[tuple[list[str], list[float]]] = []
    for row_scores, row_items in zip(score_matrix, candidate_item_ids):
        output.append(top_k_items_and_scores(row_scores, row_items, top_n=top_n))
    return output
