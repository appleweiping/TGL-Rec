"""Latency, token, and cost summaries."""

from __future__ import annotations

from statistics import median
from typing import Any


def summarize_latency(values_ms: list[float]) -> dict[str, float]:
    """Return mean/p50/p95 latency."""

    values = sorted(float(value) for value in values_ms)
    if not values:
        return {"mean_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}
    return {
        "mean_latency_ms": sum(values) / float(len(values)),
        "p50_latency_ms": float(median(values)),
        "p95_latency_ms": _percentile(values, 0.95),
    }


def summarize_token_cost(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate token and cost metadata from prediction rows."""

    prompt = 0.0
    completion = 0.0
    total = 0.0
    cost = 0.0
    for row in rows:
        usage = row.get("metadata", {}).get("llm_usage", row.get("usage", {}))
        prompt += float(usage.get("prompt_tokens", 0.0) or 0.0)
        completion += float(usage.get("completion_tokens", 0.0) or 0.0)
        total += float(usage.get("total_tokens", 0.0) or 0.0)
        cost += float(usage.get("cost", usage.get("estimated_cost", 0.0)) or 0.0)
    return {
        "completion_tokens": completion,
        "estimated_cost": cost,
        "prompt_tokens": prompt,
        "total_tokens": total,
    }


def throughput_per_second(*, item_count: int, elapsed_seconds: float) -> float:
    """Items or predictions per second."""

    if elapsed_seconds <= 0:
        return 0.0
    return float(item_count) / float(elapsed_seconds)


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)
