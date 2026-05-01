"""Token, cost, and latency aggregation for LLM diagnostics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from llm4rec.llm.base import LLMResponse


@dataclass
class CostLatencyTracker:
    """Accumulate usage metadata across LLM requests."""

    pricing: dict[str, float] = field(default_factory=dict)
    request_count: int = 0
    cache_hit_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, response: LLMResponse) -> None:
        """Record one LLM response."""

        self.request_count += 1
        if response.cache_hit:
            self.cache_hit_count += 1
        self.prompt_tokens += int(response.prompt_tokens or 0)
        self.completion_tokens += int(response.completion_tokens or 0)
        self.total_tokens += int(response.total_tokens or 0)
        self.latencies_ms.append(float(response.latency_ms or 0.0))

    def summary(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable usage summary."""

        latencies = sorted(self.latencies_ms)
        return {
            "cache_hit_count": self.cache_hit_count,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self._estimated_cost(),
            "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "prompt_tokens": self.prompt_tokens,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
        }

    def _estimated_cost(self) -> float | None:
        if not self.pricing:
            return None
        prompt_rate = float(self.pricing.get("prompt_per_1k", 0.0))
        completion_rate = float(self.pricing.get("completion_per_1k", 0.0))
        return (self.prompt_tokens / 1000.0) * prompt_rate + (
            self.completion_tokens / 1000.0
        ) * completion_rate


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight

