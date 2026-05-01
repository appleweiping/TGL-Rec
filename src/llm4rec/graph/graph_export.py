"""Graph export helpers."""

from __future__ import annotations

from typing import Any

from llm4rec.io.artifacts import write_csv_rows, write_jsonl


def export_edges_jsonl(path: str, edges: list[dict[str, Any]]) -> None:
    """Write graph edges as JSONL."""

    write_jsonl(path, edges)


def summarize_edges(
    edges: list[dict[str, Any]],
    *,
    graph_name: str,
) -> dict[str, Any]:
    """Summarize edge count, users, gaps, and buckets."""

    users = sum(int(edge.get("user_count", 0)) for edge in edges)
    weighted_sum = sum(float(edge.get("time_decayed_weight", edge.get("weight", 0.0))) for edge in edges)
    bucket_counts: dict[str, int] = {}
    for edge in edges:
        for bucket, count in dict(edge.get("bucket_counts", {})).items():
            bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + int(count)
    dominant_bucket = ""
    if bucket_counts:
        dominant_bucket = sorted(bucket_counts, key=lambda key: (-bucket_counts[key], key))[0]
    return {
        "dominant_gap_bucket": dominant_bucket,
        "edge_count": len(edges),
        "graph": graph_name,
        "time_decayed_weight_sum": weighted_sum,
        "user_count_sum": users,
    }


def write_graph_summary(path: str, rows: list[dict[str, Any]]) -> None:
    """Write graph summary CSV."""

    write_csv_rows(path, rows)
