"""Validity and hallucination metrics for generated/reranked items."""

from __future__ import annotations

from typing import Any


def prediction_item_is_valid(
    item_id: str,
    *,
    item_catalog: set[str],
    candidate_items: list[str],
    candidate_protocol: str,
) -> bool:
    """Return whether a predicted item is grounded in catalog and candidates."""

    item = str(item_id)
    if item not in item_catalog:
        return False
    if candidate_protocol != "no_candidates" and candidate_items:
        return item in {str(candidate) for candidate in candidate_items}
    return True


def aggregate_validity_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    item_catalog: set[str],
    candidate_protocol: str,
) -> dict[str, float]:
    """Compute aggregate validity and hallucination rates."""

    total = 0
    invalid = 0
    for row in prediction_rows:
        candidates = [str(item) for item in row.get("candidate_items", [])]
        for item_id in row.get("predicted_items", []):
            total += 1
            if not prediction_item_is_valid(
                str(item_id),
                item_catalog=item_catalog,
                candidate_items=candidates,
                candidate_protocol=candidate_protocol,
            ):
                invalid += 1
    if total == 0:
        return {"validity_rate": 1.0, "hallucination_rate": 0.0}
    hallucination_rate = invalid / float(total)
    return {
        "hallucination_rate": hallucination_rate,
        "validity_rate": 1.0 - hallucination_rate,
    }
