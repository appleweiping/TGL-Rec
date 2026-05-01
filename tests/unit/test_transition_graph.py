from __future__ import annotations

from llm4rec.graph.transition_graph import build_transition_edges


def test_transition_graph_counts_directed_edges() -> None:
    rows = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2},
        {"user_id": "u1", "item_id": "i3", "timestamp": 3},
        {"user_id": "u2", "item_id": "i1", "timestamp": 1},
        {"user_id": "u2", "item_id": "i2", "timestamp": 4},
    ]
    edges = build_transition_edges(rows)
    edge = next(row for row in edges if row["source_item"] == "i1" and row["target_item"] == "i2")
    assert edge["count"] == 2
    assert edge["user_count"] == 2
    assert edge["bucket_counts"]["same_session"] == 2
