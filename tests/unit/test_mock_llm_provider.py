import pytest

from llm4rec.llm.base import LLMRequest
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.llm.safety import LLMProviderSafetyError
from llm4rec.prompts.parsers import parse_llm_response


def _request(**metadata):
    return LLMRequest(
        prompt="Rank candidates",
        prompt_version="v1",
        candidate_item_ids=["i1", "i2", "i3"],
        provider="mock",
        model="mock-llm",
        metadata={"run_mode": "diagnostic_mock", **metadata},
    )


def test_mock_provider_transition_mode_uses_scores():
    provider = MockLLMProvider(mode="transition_aware", run_mode="diagnostic_mock")
    response = provider.generate(_request(transition_scores={"i2": 10.0, "i1": 1.0}))
    parsed = parse_llm_response(response.raw_output, candidate_items=["i1", "i2", "i3"])

    assert parsed.ranked_item_ids[0] == "i2"
    assert response.total_tokens > 0


def test_mock_provider_hallucinating_mode_for_tests():
    provider = MockLLMProvider(mode="hallucinating", run_mode="diagnostic_mock")
    response = provider.generate(_request())
    parsed = parse_llm_response(response.raw_output, candidate_items=["i1", "i2", "i3"])

    assert parsed.invalid_item_ids == ["not_in_candidates"]


def test_mock_provider_forbidden_in_reportable_mode():
    with pytest.raises(LLMProviderSafetyError):
        MockLLMProvider(mode="identity", run_mode="reportable")

