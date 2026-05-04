import json

import pytest

from llm4rec.llm.base import LLMRequest
from llm4rec.llm.deepseek_provider import DeepSeekProviderConfig, DeepSeekV4FlashProvider
from llm4rec.llm.safety import LLMProviderSafetyError


def test_deepseek_payload_is_openai_compatible_and_json_mode():
    provider = DeepSeekV4FlashProvider(
        DeepSeekProviderConfig(model="deepseek-v4-flash"),
        allow_api_calls=True,
    )
    request = LLMRequest(
        prompt="rank",
        prompt_version="history_only",
        candidate_item_ids=["i1"],
        provider="deepseek",
        model="deepseek-v4-flash",
    )

    payload = provider.payload_for(request)

    assert payload["model"] == "deepseek-v4-flash"
    assert payload["messages"][-1] == {"role": "user", "content": "rank"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False
    assert payload["thinking"] == {"type": "disabled"}


def test_deepseek_provider_requires_api_opt_in(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    provider = DeepSeekV4FlashProvider(allow_api_calls=False)
    request = LLMRequest(
        prompt="rank",
        prompt_version="history_only",
        candidate_item_ids=["i1"],
        provider="deepseek",
        model="deepseek-v4-flash",
    )

    with pytest.raises(LLMProviderSafetyError):
        provider.generate(request)


def test_deepseek_response_conversion_does_not_store_headers():
    provider = DeepSeekV4FlashProvider()
    response = provider._to_response(
        {
            "id": "abc",
            "choices": [{"message": {"content": json.dumps({"ranked_item_ids": ["i1"]})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
        latency_ms=10,
    )

    assert response.raw_output == '{"ranked_item_ids": ["i1"]}'
    assert response.total_tokens == 5
    assert "authorization" not in response.metadata
