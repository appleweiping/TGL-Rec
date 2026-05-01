"""Evidence construction from train-only temporal graph artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from llm4rec.evidence.base import Evidence
from llm4rec.graph.time_window_graph import build_time_window_edges
from llm4rec.graph.transition_graph import build_transition_edges
from llm4rec.io.artifacts import write_jsonl


def build_temporal_graph_artifacts(
    *,
    train_interactions: list[dict[str, Any]],
    output_dir: str | Path,
    window_seconds: int | float,
    candidate_protocol: str,
) -> dict[str, Any]:
    """Build train-only temporal graph artifacts for smoke runs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transition_edges = build_transition_edges(train_interactions)
    time_window_edges = build_time_window_edges(
        train_interactions,
        window_seconds=window_seconds,
        directed=True,
        weight_mode="time_decay",
        half_life_seconds=window_seconds,
    )
    transition_path = output / "transition_edges.jsonl"
    time_window_path = output / "time_window_edges.jsonl"
    write_jsonl(transition_path, transition_edges)
    write_jsonl(time_window_path, time_window_edges)
    return {
        "candidate_protocol": candidate_protocol,
        "constructed_from": "train_only",
        "time_window_edges": time_window_edges,
        "time_window_path": str(time_window_path),
        "transition_edges": transition_edges,
        "transition_path": str(transition_path),
        "window_seconds": window_seconds,
    }


def transition_edge_to_evidence(
    edge: dict[str, Any],
    *,
    graph_artifact: str,
    candidate_protocol: str,
    constructed_from: str = "train_only",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Convert a directed transition edge into evidence."""

    source = str(edge["source_item"])
    target = str(edge["target_item"])
    gap_bucket = _dominant_bucket(edge.get("bucket_counts", {}))
    stats = {
        "transition_count": int(edge.get("count", 0)),
        "user_count": int(edge.get("user_count", 0)),
        "time_window_score": None,
        "semantic_similarity": None,
        "time_decayed_weight": None,
    }
    timestamp_info = {
        "source_timestamp": None,
        "target_timestamp": None,
        "mean_gap_seconds": edge.get("mean_time_gap"),
        "median_gap_seconds": edge.get("median_time_gap"),
        "gap_bucket": gap_bucket,
    }
    return Evidence(
        evidence_id=_evidence_id("transition", source, target, stats, timestamp_info),
        evidence_type="transition",
        source_item=source,
        target_item=target,
        support_items=[source, target],
        timestamp_info=timestamp_info,
        stats=stats,
        text="",
        provenance=_provenance(
            graph_artifact=graph_artifact,
            candidate_protocol=candidate_protocol,
            constructed_from=constructed_from,
        ),
        metadata=dict(metadata or {}),
    )


def time_window_edge_to_evidence(
    edge: dict[str, Any],
    *,
    graph_artifact: str,
    candidate_protocol: str,
    constructed_from: str = "train_only",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Convert a time-window edge into evidence."""

    source = str(edge["source_item"])
    target = str(edge["target_item"])
    gap_bucket = _dominant_bucket(edge.get("bucket_counts", {}))
    weight = float(edge.get("weight", edge.get("time_decayed_weight", 0.0)) or 0.0)
    stats = {
        "transition_count": int(edge.get("count", 0)),
        "user_count": int(edge.get("user_count", 0)),
        "time_window_score": weight,
        "semantic_similarity": None,
        "time_decayed_weight": float(edge.get("time_decayed_weight", weight) or 0.0),
    }
    timestamp_info = {
        "source_timestamp": None,
        "target_timestamp": None,
        "mean_gap_seconds": edge.get("mean_time_gap"),
        "median_gap_seconds": edge.get("median_time_gap"),
        "gap_bucket": gap_bucket,
    }
    return Evidence(
        evidence_id=_evidence_id("time_window", source, target, stats, timestamp_info),
        evidence_type="time_window",
        source_item=source,
        target_item=target,
        support_items=[source, target],
        timestamp_info=timestamp_info,
        stats=stats,
        text="",
        provenance=_provenance(
            graph_artifact=graph_artifact,
            candidate_protocol=candidate_protocol,
            constructed_from=constructed_from,
        ),
        metadata={**dict(metadata or {}), "window_seconds": edge.get("window_seconds")},
    )


def make_history_evidence(
    *,
    source_item: str,
    target_item: str,
    recent_rank: int,
    candidate_protocol: str,
    constructed_from: str = "train_only",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Construct recent-history evidence for a candidate without target labels."""

    recency = 1.0 / float(max(1, recent_rank))
    stats = {
        "transition_count": 0,
        "user_count": 1,
        "time_window_score": None,
        "semantic_similarity": None,
        "time_decayed_weight": None,
        "recent_signal": recency,
    }
    return Evidence(
        evidence_id=_evidence_id("history", source_item, target_item, stats, {"gap_bucket": "history"}),
        evidence_type="history",
        source_item=str(source_item),
        target_item=str(target_item),
        support_items=[str(source_item), str(target_item)],
        timestamp_info={
            "source_timestamp": None,
            "target_timestamp": None,
            "mean_gap_seconds": None,
            "median_gap_seconds": None,
            "gap_bucket": "history",
        },
        stats=stats,
        text="",
        provenance=_provenance(
            graph_artifact="user_history",
            candidate_protocol=candidate_protocol,
            constructed_from=constructed_from,
        ),
        metadata=dict(metadata or {}),
    )


def make_semantic_evidence(
    *,
    source_item: str,
    target_item: str,
    similarity: float,
    candidate_protocol: str,
    constructed_from: str = "train_only",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Construct metadata-based semantic evidence."""

    stats = {
        "transition_count": 0,
        "user_count": 0,
        "time_window_score": None,
        "semantic_similarity": float(similarity),
        "time_decayed_weight": None,
    }
    return Evidence(
        evidence_id=_evidence_id("semantic", source_item, target_item, stats, {"gap_bucket": "metadata"}),
        evidence_type="semantic",
        source_item=str(source_item),
        target_item=str(target_item),
        support_items=[str(source_item), str(target_item)],
        timestamp_info={
            "source_timestamp": None,
            "target_timestamp": None,
            "mean_gap_seconds": None,
            "median_gap_seconds": None,
            "gap_bucket": "metadata",
        },
        stats=stats,
        text="",
        provenance=_provenance(
            graph_artifact="item_text_metadata",
            candidate_protocol=candidate_protocol,
            constructed_from=constructed_from,
        ),
        metadata=dict(metadata or {}),
    )


def make_contrastive_evidence(
    *,
    source_item: str,
    target_item: str,
    transition_count: int,
    semantic_similarity: float,
    candidate_protocol: str,
    constructed_from: str = "train_only",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Construct contrastive transition-vs-semantic evidence."""

    stats = {
        "transition_count": int(transition_count),
        "user_count": 0,
        "time_window_score": None,
        "semantic_similarity": float(semantic_similarity),
        "time_decayed_weight": None,
    }
    return Evidence(
        evidence_id=_evidence_id("contrastive", source_item, target_item, stats, {"gap_bucket": "contrastive"}),
        evidence_type="contrastive",
        source_item=str(source_item),
        target_item=str(target_item),
        support_items=[str(source_item), str(target_item)],
        timestamp_info={
            "source_timestamp": None,
            "target_timestamp": None,
            "mean_gap_seconds": None,
            "median_gap_seconds": None,
            "gap_bucket": "contrastive",
        },
        stats=stats,
        text="",
        provenance=_provenance(
            graph_artifact="transition_edges_plus_item_text_metadata",
            candidate_protocol=candidate_protocol,
            constructed_from=constructed_from,
        ),
        metadata=dict(metadata or {}),
    )


def make_user_drift_evidence(
    *,
    source_item: str,
    target_item: str,
    candidate_protocol: str,
    drift_from: str | None,
    drift_to: str | None,
    constructed_from: str = "train_only",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Construct user-drift evidence from history categories only."""

    stats = {
        "transition_count": 0,
        "user_count": 1,
        "time_window_score": None,
        "semantic_similarity": None,
        "time_decayed_weight": None,
        "recent_signal": 1.0,
    }
    return Evidence(
        evidence_id=_evidence_id("user_drift", source_item, target_item, stats, {"gap_bucket": "drift"}),
        evidence_type="user_drift",
        source_item=str(source_item),
        target_item=str(target_item),
        support_items=[str(source_item), str(target_item)],
        timestamp_info={
            "source_timestamp": None,
            "target_timestamp": None,
            "mean_gap_seconds": None,
            "median_gap_seconds": None,
            "gap_bucket": "drift",
        },
        stats=stats,
        text="",
        provenance=_provenance(
            graph_artifact="user_history_categories",
            candidate_protocol=candidate_protocol,
            constructed_from=constructed_from,
        ),
        metadata={**dict(metadata or {}), "drift_from": drift_from, "drift_to": drift_to},
    )


def _provenance(
    *,
    graph_artifact: str,
    candidate_protocol: str,
    constructed_from: str,
) -> dict[str, Any]:
    return {
        "graph_artifact": str(graph_artifact),
        "split": "train",
        "candidate_protocol": str(candidate_protocol),
        "constructed_from": str(constructed_from),
    }


def _dominant_bucket(bucket_counts: dict[str, Any]) -> str | None:
    if not bucket_counts:
        return None
    return sorted(bucket_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _evidence_id(
    evidence_type: str,
    source_item: str | None,
    target_item: str | None,
    stats: dict[str, Any],
    timestamp_info: dict[str, Any],
) -> str:
    import json

    payload = json.dumps(
        {
            "evidence_type": evidence_type,
            "source_item": source_item,
            "target_item": target_item,
            "stats": stats,
            "timestamp_info": timestamp_info,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return "ev_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
