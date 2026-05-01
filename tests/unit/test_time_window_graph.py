from __future__ import annotations

import math

from llm4rec.graph.edge_weights import exponential_decay_weight
from llm4rec.graph.time_window_graph import build_time_window_edges


def test_exponential_decay_weight_half_life() -> None:
    assert exponential_decay_weight(0, half_life_seconds=10) == 1.0
    assert math.isclose(exponential_decay_weight(10, half_life_seconds=10), 0.5)


def test_time_window_graph_builds_undirected_edges() -> None:
    rows = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 3},
        {"user_id": "u1", "item_id": "i3", "timestamp": 20},
    ]
    edges = build_time_window_edges(rows, window_seconds=5, directed=False, weight_mode="count")
    assert len(edges) == 1
    assert edges[0]["source_item"] == "i1"
    assert edges[0]["target_item"] == "i2"
    assert edges[0]["weight"] == 1.0


def test_time_window_graph_builds_decay_weights() -> None:
    rows = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 11},
    ]
    edges = build_time_window_edges(
        rows,
        window_seconds=20,
        directed=True,
        weight_mode="time_decay",
        half_life_seconds=10,
    )
    assert edges[0]["source_item"] == "i1"
    assert math.isclose(edges[0]["weight"], 0.5)
