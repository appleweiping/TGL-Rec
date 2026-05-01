"""Rule-based grounding checks for Phase 3A LLM diagnostic explanations."""

from __future__ import annotations

from typing import Any


def build_edge_index(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index directed or undirected edge rows by source/target item IDs."""

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        source = str(edge.get("source_item"))
        target = str(edge.get("target_item"))
        index[(source, target)] = edge
        if not bool(edge.get("directed", True)):
            index[(target, source)] = edge
    return index


def evaluate_evidence_grounding(
    evidence_items: list[dict[str, Any]],
    *,
    history_items: list[str],
    candidate_items: list[str],
    transition_edges: dict[tuple[str, str], dict[str, Any]],
    time_window_edges: dict[tuple[str, str], dict[str, Any]],
    time_bucket_by_pair: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    """Evaluate whether evidence references actual history/candidate/graph facts."""

    allowed_items = {str(item) for item in history_items} | {str(item) for item in candidate_items}
    bucket_by_pair = time_bucket_by_pair or {}
    results = []
    for item in evidence_items:
        result = ground_evidence_item(
            item,
            allowed_items=allowed_items,
            transition_edges=transition_edges,
            time_window_edges=time_window_edges,
            time_bucket_by_pair=bucket_by_pair,
        )
        results.append(result)
    grounded = sum(1 for row in results if row["grounded"])
    total = len(results)
    return {
        "evidence_count": total,
        "evidence_grounding_rate": grounded / float(total or 1),
        "grounded_evidence_count": grounded,
        "items": results,
        "semantic_evidence_usage": any(row.get("type") == "semantic" for row in results),
        "time_evidence_usage": any(row.get("type") in {"time_gap", "time_bucket", "time_window"} for row in results),
        "transition_evidence_usage": any(row.get("type") == "transition" for row in results),
    }


def ground_evidence_item(
    item: dict[str, Any],
    *,
    allowed_items: set[str],
    transition_edges: dict[tuple[str, str], dict[str, Any]],
    time_window_edges: dict[tuple[str, str], dict[str, Any]],
    time_bucket_by_pair: dict[tuple[str, str], str],
) -> dict[str, Any]:
    """Ground one evidence item against known item IDs and graph edges."""

    evidence_type = str(item.get("type", "semantic"))
    source = item.get("source_item")
    target = item.get("target_item")
    source_id = None if source is None else str(source)
    target_id = None if target is None else str(target)
    text = str(item.get("text", ""))
    item_ids_ok = all(
        value in allowed_items
        for value in (source_id, target_id)
        if value not in (None, "")
    )
    edge_ok = True
    bucket_ok = True
    if evidence_type == "transition":
        edge_ok = source_id is not None and target_id is not None and (source_id, target_id) in transition_edges
    elif evidence_type == "time_window":
        edge_ok = source_id is not None and target_id is not None and (
            (source_id, target_id) in time_window_edges or (target_id, source_id) in time_window_edges
        )
    elif evidence_type in {"time_bucket", "time_gap"} and source_id is not None and target_id is not None:
        expected = time_bucket_by_pair.get((source_id, target_id))
        bucket_ok = expected is None or expected in text
    grounded = bool(item_ids_ok and edge_ok and bucket_ok)
    return {
        "bucket_ok": bucket_ok,
        "edge_ok": edge_ok,
        "grounded": grounded,
        "item_ids_ok": item_ids_ok,
        "source_item": source_id,
        "target_item": target_id,
        "text": text,
        "type": evidence_type,
    }


def aggregate_grounding(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate grounding metadata stored on prediction rows."""

    if not rows:
        return {
            "evidence_grounding_rate": 0.0,
            "semantic_evidence_usage_rate": 0.0,
            "time_evidence_usage_rate": 0.0,
            "transition_evidence_usage_rate": 0.0,
        }
    return {
        "evidence_grounding_rate": sum(
            float(row.get("metadata", {}).get("grounding", {}).get("evidence_grounding_rate", 0.0))
            for row in rows
        )
        / float(len(rows)),
        "semantic_evidence_usage_rate": sum(
            1.0
            for row in rows
            if row.get("metadata", {}).get("grounding", {}).get("semantic_evidence_usage")
        )
        / float(len(rows)),
        "time_evidence_usage_rate": sum(
            1.0
            for row in rows
            if row.get("metadata", {}).get("grounding", {}).get("time_evidence_usage")
        )
        / float(len(rows)),
        "transition_evidence_usage_rate": sum(
            1.0
            for row in rows
            if row.get("metadata", {}).get("grounding", {}).get("transition_evidence_usage")
        )
        / float(len(rows)),
    }

