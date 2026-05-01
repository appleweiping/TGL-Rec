import pytest

from llm4rec.llm.cost_estimator import (
    assert_within_call_cap,
    build_cost_preflight,
    estimate_token_count,
)


def test_estimate_token_count_is_positive_for_text():
    assert estimate_token_count("one two three four") >= 4


def test_cost_preflight_counts_prompts_and_completion_budget():
    preflight = build_cost_preflight(
        prompts=["hello world", "another prompt"],
        number_of_cases=1,
        prompt_variants=["history_only", "history_with_order"],
        max_tokens=32,
        max_api_calls=125,
        cache_enabled=True,
        cache_policy="read_write",
        model_name="test-model",
        run_dir="outputs/runs/test",
    )
    assert preflight.estimated_api_calls == 2
    assert preflight.estimated_completion_tokens == 64
    assert preflight.to_dict()["model_name"] == "test-model"


def test_cost_preflight_hard_cap_raises():
    preflight = build_cost_preflight(
        prompts=["x"] * 126,
        number_of_cases=26,
        prompt_variants=["history_only"],
        max_tokens=16,
        max_api_calls=125,
        cache_enabled=False,
        cache_policy="disabled",
        model_name="test-model",
        run_dir="outputs/runs/test",
    )
    with pytest.raises(ValueError, match="exceed max_api_calls"):
        assert_within_call_cap(preflight)
