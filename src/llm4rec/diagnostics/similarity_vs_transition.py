"""Similarity-vs-transition diagnostic artifact."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm4rec.data.text_fields import item_text
from llm4rec.graph.time_window_graph import build_time_window_edges
from llm4rec.graph.transition_graph import build_transition_edges
from llm4rec.rankers.bm25 import tokenize


def build_similarity_vs_transition_artifact(
    interactions: list[dict[str, Any]],
    item_records: list[dict[str, Any]],
    *,
    window_seconds: int | float,
    top_k: int = 50,
) -> dict[str, Any]:
    """Compare item text overlap with transition and time-window strengths."""

    item_by_id = {str(row["item_id"]): row for row in item_records}
    tokens_by_item = {
        item_id: set(tokenize(item_text(row))) for item_id, row in item_by_id.items()
    }
    transition_edges = build_transition_edges(interactions)
    transition_scores = {
        (str(edge["source_item"]), str(edge["target_item"])): float(edge["count"])
        for edge in transition_edges
    }
    window_edges = build_time_window_edges(
        interactions,
        window_seconds=window_seconds,
        directed=False,
        weight_mode="count",
    )
    window_scores = {
        tuple(sorted((str(edge["source_item"]), str(edge["target_item"])))): float(edge["weight"])
        for edge in window_edges
    }
    rows: list[dict[str, Any]] = []
    item_ids = sorted(item_by_id)
    for source in item_ids:
        for target in item_ids:
            if source == target:
                continue
            source_tokens = tokens_by_item.get(source, set())
            target_tokens = tokens_by_item.get(target, set())
            union = source_tokens | target_tokens
            similarity = 0.0 if not union else len(source_tokens & target_tokens) / float(len(union))
            transition_strength = transition_scores.get((source, target), 0.0)
            window_strength = window_scores.get(tuple(sorted((source, target))), 0.0)
            if similarity == 0.0 and transition_strength == 0.0 and window_strength == 0.0:
                continue
            source_category = item_by_id[source].get("category")
            target_category = item_by_id[target].get("category")
            edge = _find_transition_edge(transition_edges, source, target)
            rows.append(
                {
                    "category_relation": "same" if source_category == target_category else "different",
                    "source_item": source,
                    "target_item": target,
                    "text_similarity": similarity,
                    "time_gap_bucket_distribution": edge.get("bucket_counts", {}) if edge else {},
                    "transition_strength": transition_strength,
                    "window_strength": window_strength,
                }
            )
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["transition_strength"]),
            -float(row["window_strength"]),
            -float(row["text_similarity"]),
            row["source_item"],
            row["target_item"],
        ),
    )[:top_k]
    summary = _summarize(rows)
    return {
        "pairs": rows,
        "summary": summary,
        "window_seconds": float(window_seconds),
    }


def _find_transition_edge(
    edges: list[dict[str, Any]],
    source: str,
    target: str,
) -> dict[str, Any] | None:
    for edge in edges:
        if str(edge["source_item"]) == source and str(edge["target_item"]) == target:
            return edge
    return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        similarity = float(row["text_similarity"])
        transition = float(row["transition_strength"])
        if transition > 0 and similarity > 0:
            counts["semantic_and_transition"] += 1
        elif transition > 0:
            counts["transition_only"] += 1
        elif similarity > 0:
            counts["semantic_only"] += 1
        else:
            counts["window_only"] += 1
    return dict(sorted(counts.items()))
