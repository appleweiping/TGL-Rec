"""Beyond-accuracy diversity and coverage metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Any


def item_coverage(prediction_rows: list[dict[str, Any]], *, k: int | None = None) -> int:
    """Count unique predicted items."""

    return len(_predicted_set(prediction_rows, k=k))


def catalog_coverage(prediction_rows: list[dict[str, Any]], item_catalog: set[str], *, k: int | None = None) -> float:
    """Fraction of catalog items appearing in predictions."""

    if not item_catalog:
        return 0.0
    return len(_predicted_set(prediction_rows, k=k) & {str(item) for item in item_catalog}) / float(len(item_catalog))


def intra_list_diversity(
    predicted_items: list[str],
    *,
    item_features: dict[str, set[str]] | None = None,
    k: int | None = None,
) -> float:
    """Average pairwise dissimilarity within one recommendation list."""

    items = [str(item) for item in predicted_items[:k]]
    if len(items) < 2:
        return 0.0
    values: list[float] = []
    for left, right in combinations(items, 2):
        values.append(1.0 - _jaccard(item_features.get(left, {left}) if item_features else {left}, item_features.get(right, {right}) if item_features else {right}))
    return sum(values) / float(len(values))


def aggregate_intra_list_diversity(
    prediction_rows: list[dict[str, Any]],
    *,
    item_features: dict[str, set[str]] | None = None,
    k: int | None = None,
) -> float:
    """Mean intra-list diversity over prediction rows."""

    if not prediction_rows:
        return 0.0
    return sum(
        intra_list_diversity(row.get("predicted_items", []), item_features=item_features, k=k)
        for row in prediction_rows
    ) / float(len(prediction_rows))


def _predicted_set(prediction_rows: list[dict[str, Any]], *, k: int | None) -> set[str]:
    output: set[str] = set()
    for row in prediction_rows:
        items = [str(item) for item in row.get("predicted_items", [])]
        output.update(items[:k] if k is not None else items)
    return output


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / float(len(union))
