"""Time-window item co-occurrence graph construction."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from llm4rec.data.time_features import bucket_time_gap
from llm4rec.graph.edge_weights import exponential_decay_weight


def build_time_window_edges(
    interactions: list[dict[str, Any]],
    *,
    window_seconds: int | float,
    directed: bool = False,
    weight_mode: str = "count",
    half_life_seconds: int | float | None = None,
) -> list[dict[str, Any]]:
    """Build item pairs that occur within a configured time window for one user."""

    window = float(window_seconds)
    if window <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")
    if weight_mode not in {"count", "time_decay"}:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")
    if weight_mode == "time_decay" and half_life_seconds is None:
        half_life_seconds = window

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for user_id in sorted(by_user):
        rows = sorted(
            by_user[user_id],
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            ),
        )
        for left_index, left in enumerate(rows):
            left_ts = left.get("timestamp")
            if left_ts is None:
                continue
            for right in rows[left_index + 1 :]:
                right_ts = right.get("timestamp")
                if right_ts is None:
                    continue
                gap = float(right_ts) - float(left_ts)
                if gap < 0:
                    continue
                if gap > window:
                    break
                pairs = [(str(left["item_id"]), str(right["item_id"]))]
                if not directed:
                    source, target = sorted(pairs[0])
                    pairs = [(source, target)]
                for source, target in pairs:
                    weight = 1.0
                    if weight_mode == "time_decay":
                        weight = exponential_decay_weight(
                            gap,
                            half_life_seconds=float(half_life_seconds or window),
                        )
                    key = (source, target)
                    stats = edges.setdefault(
                        key,
                        {
                            "bucket_counts": defaultdict(int),
                            "count": 0,
                            "gaps": [],
                            "source_item": source,
                            "target_item": target,
                            "users": set(),
                            "weight": 0.0,
                        },
                    )
                    stats["count"] += 1
                    stats["gaps"].append(gap)
                    stats["weight"] += weight
                    stats["users"].add(user_id)
                    stats["bucket_counts"][bucket_time_gap(gap)] += 1

    rows: list[dict[str, Any]] = []
    for stats in edges.values():
        gaps = list(stats["gaps"])
        rows.append(
            {
                "bucket_counts": dict(sorted(stats["bucket_counts"].items())),
                "count": int(stats["count"]),
                "directed": bool(directed),
                "mean_time_gap": float(statistics.fmean(gaps)) if gaps else None,
                "median_time_gap": float(statistics.median(gaps)) if gaps else None,
                "source_item": str(stats["source_item"]),
                "target_item": str(stats["target_item"]),
                "time_decayed_weight": float(stats["weight"]),
                "user_count": len(stats["users"]),
                "weight": float(stats["weight"]),
                "weight_mode": weight_mode,
                "window_seconds": window,
            }
        )
    return sorted(rows, key=lambda row: (-row["weight"], row["source_item"], row["target_item"]))
