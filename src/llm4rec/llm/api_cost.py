"""API token and cost accounting for DeepSeek experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class TokenPricing:
    """Per-1M-token pricing in USD."""

    input_cache_hit_per_1m: float
    input_cache_miss_per_1m: float
    output_per_1m: float
    source: str = ""


DEEPSEEK_V4_FLASH_PRICING = TokenPricing(
    input_cache_hit_per_1m=0.028,
    input_cache_miss_per_1m=0.14,
    output_per_1m=0.28,
    source="https://api-docs.deepseek.com/quick_start/pricing",
)


def estimate_request_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit: bool,
    pricing: TokenPricing,
) -> float:
    """Estimate cost for one request."""

    input_rate = pricing.input_cache_hit_per_1m if cache_hit else pricing.input_cache_miss_per_1m
    return (int(prompt_tokens) / 1_000_000.0) * input_rate + (
        int(completion_tokens) / 1_000_000.0
    ) * pricing.output_per_1m


def summarize_cost_latency(rows: list[dict[str, Any]], *, pricing: TokenPricing) -> dict[str, Any]:
    """Summarize request usage, latency, cache, throughput, and estimated cost."""

    if not rows:
        return {
            "cache_hit_rate": 0.0,
            "estimated_cost": 0.0,
            "latency_mean": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "requests": 0,
            "requests_per_minute": 0.0,
            "total_tokens": 0,
        }
    latencies = sorted(float(row.get("latency_ms", 0.0) or 0.0) for row in rows)
    total_runtime_seconds = sum(latencies) / 1000.0
    cache_hits = sum(1 for row in rows if bool(row.get("cache_hit", False)))
    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in rows)
    estimated_cost = sum(
        estimate_request_cost_usd(
            prompt_tokens=int(row.get("prompt_tokens", 0) or 0),
            completion_tokens=int(row.get("completion_tokens", 0) or 0),
            cache_hit=bool(row.get("cache_hit", False)),
            pricing=pricing,
        )
        for row in rows
    )
    return {
        "cache_hit_rate": cache_hits / float(len(rows)),
        "completion_tokens": completion_tokens,
        "estimated_cost": estimated_cost,
        "latency_mean": mean(latencies),
        "latency_p50": median(latencies),
        "latency_p95": _percentile(latencies, 0.95),
        "pricing_source": pricing.source,
        "prompt_tokens": prompt_tokens,
        "requests": len(rows),
        "requests_per_minute": len(rows) / (total_runtime_seconds / 60.0)
        if total_runtime_seconds > 0
        else 0.0,
        "total_tokens": total_tokens,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return values[index]
