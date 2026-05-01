from llm4rec.metrics.efficiency import summarize_latency, summarize_token_cost, throughput_per_second


def test_efficiency_summaries():
    latency = summarize_latency([10, 20, 30])
    assert latency["p50_latency_ms"] == 20
    tokens = summarize_token_cost([{"metadata": {"llm_usage": {"prompt_tokens": 3, "completion_tokens": 2}}}])
    assert tokens["prompt_tokens"] == 3
    assert throughput_per_second(item_count=10, elapsed_seconds=2) == 5
