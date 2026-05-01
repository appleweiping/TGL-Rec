"""Transition diagnostic metrics."""

from __future__ import annotations

from typing import Any


def transition_coverage(
    transition_edges: list[dict[str, Any]],
    *,
    item_catalog: set[str],
) -> float:
    """Fraction of catalog items participating in at least one transition edge."""

    if not item_catalog:
        return 0.0
    touched = {
        str(edge[field])
        for edge in transition_edges
        for field in ("source_item", "target_item")
        if str(edge[field]) in item_catalog
    }
    return len(touched) / float(len(item_catalog))


def mean_transition_count(transition_edges: list[dict[str, Any]]) -> float:
    """Mean edge support count."""

    if not transition_edges:
        return 0.0
    return sum(float(edge.get("count", 0.0)) for edge in transition_edges) / float(len(transition_edges))
