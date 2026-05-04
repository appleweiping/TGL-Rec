from llm4rec.llm.rate_limit import AdaptiveConcurrencyController, RateLimitConfig


def test_adaptive_concurrency_reduces_on_429():
    controller = AdaptiveConcurrencyController(
        RateLimitConfig(max_concurrency=32, min_concurrency=4, adaptive_concurrency=True)
    )

    controller.record_status(429)

    assert controller.current_limit == 16
    assert controller.report()["rate_limit_events"] == 1


def test_backoff_respects_max_without_jitter():
    controller = AdaptiveConcurrencyController(
        RateLimitConfig(backoff_initial_seconds=2, backoff_max_seconds=5, jitter=False)
    )

    assert controller.backoff_seconds(4) == 5
