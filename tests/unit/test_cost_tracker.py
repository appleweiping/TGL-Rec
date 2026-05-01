from llm4rec.llm.base import LLMResponse
from llm4rec.llm.cost_tracker import CostLatencyTracker


def test_cost_tracker_aggregates_tokens_latency_and_cost():
    tracker = CostLatencyTracker(pricing={"prompt_per_1k": 0.01, "completion_per_1k": 0.02})
    tracker.record(
        LLMResponse(
            raw_output="{}",
            provider="mock",
            model="mock",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=10,
        )
    )
    tracker.record(
        LLMResponse(
            raw_output="{}",
            provider="mock",
            model="mock",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=20,
            cache_hit=True,
        )
    )
    summary = tracker.summary()

    assert summary["request_count"] == 2
    assert summary["cache_hit_count"] == 1
    assert summary["total_tokens"] == 300
    assert summary["estimated_cost"] == 0.004
    assert summary["p50_latency_ms"] == 15

