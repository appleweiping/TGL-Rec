"""Directed item-transition graph construction."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from llm4rec.data.time_features import bucket_time_gap


def build_transition_edges(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build directed item_i -> item_j edges from consecutive user interactions."""

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)

    source_transition_counts: Counter[str] = Counter()
    target_transition_counts: Counter[str] = Counter()
    raw_edges: dict[tuple[str, str], dict[str, Any]] = {}
    total_transitions = 0
    for user_id in sorted(by_user):
        rows = sorted(
            by_user[user_id],
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            ),
        )
        for left, right in zip(rows, rows[1:]):
            source = str(left["item_id"])
            target = str(right["item_id"])
            left_ts = left.get("timestamp")
            right_ts = right.get("timestamp")
            gap = None if left_ts is None or right_ts is None else float(right_ts) - float(left_ts)
            if gap is not None and gap < 0:
                continue
            key = (source, target)
            stats = raw_edges.setdefault(
                key,
                {
                    "bucket_counts": Counter(),
                    "count": 0,
                    "gaps": [],
                    "source_item": source,
                    "target_item": target,
                    "users": set(),
                },
            )
            stats["count"] += 1
            stats["users"].add(user_id)
            if gap is not None:
                stats["gaps"].append(gap)
            stats["bucket_counts"][bucket_time_gap(gap)] += 1
            source_transition_counts[source] += 1
            target_transition_counts[target] += 1
            total_transitions += 1

    edges: list[dict[str, Any]] = []
    for (source, target), stats in raw_edges.items():
        gaps = list(stats["gaps"])
        bucket_counts = dict(sorted(stats["bucket_counts"].items()))
        count = int(stats["count"])
        source_count = int(source_transition_counts[source])
        target_count = int(target_transition_counts[target])
        transition_probability = count / float(source_count or 1)
        target_probability = target_count / float(total_transitions or 1)
        lift = transition_probability / target_probability if target_probability > 0 else 0.0
        reverse_count = int(raw_edges.get((target, source), {}).get("count", 0))
        direction_asymmetry = (
            (count - reverse_count) / float(count + reverse_count)
            if count + reverse_count > 0
            else 0.0
        )
        edges.append(
            {
                "bucket_counts": bucket_counts,
                "count": count,
                "direction_asymmetry": direction_asymmetry,
                "lift": lift,
                "mean_time_gap": float(statistics.fmean(gaps)) if gaps else None,
                "median_time_gap": float(statistics.median(gaps)) if gaps else None,
                "pmi": 0.0 if lift <= 0.0 else _safe_log(lift),
                "source_item": str(stats["source_item"]),
                "source_transition_count": source_count,
                "target_item": str(stats["target_item"]),
                "target_transition_count": target_count,
                "total_transition_count": total_transitions,
                "transition_probability": transition_probability,
                "user_count": len(stats["users"]),
            }
        )
    return sorted(edges, key=lambda row: (-row["count"], row["source_item"], row["target_item"]))


def _safe_log(value: float) -> float:
    import math

    return float(math.log(value))
