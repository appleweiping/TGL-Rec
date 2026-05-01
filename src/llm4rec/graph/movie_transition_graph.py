"""MovieLens diagnostic transition-graph helpers."""

from __future__ import annotations

from typing import Any

from llm4rec.graph.edge_weights import exponential_decay_weight
from llm4rec.graph.transition_graph import build_transition_edges


def build_movie_transition_edges(
    interactions: list[dict[str, Any]],
    *,
    half_life_seconds: int | float,
) -> list[dict[str, Any]]:
    """Build directed transitions with an additional time-decay weight."""

    edges = build_transition_edges(interactions)
    output: list[dict[str, Any]] = []
    for edge in edges:
        row = dict(edge)
        mean_gap = row.get("mean_time_gap")
        if mean_gap is None:
            row["time_decayed_weight"] = float(row["count"])
        else:
            row["time_decayed_weight"] = float(row["count"]) * exponential_decay_weight(
                float(mean_gap),
                half_life_seconds=half_life_seconds,
            )
        output.append(row)
    return output
