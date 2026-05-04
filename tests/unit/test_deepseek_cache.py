from llm4rec.llm.api_cache import APICache
from llm4rec.llm.base import LLMRequest, LLMResponse


def test_api_cache_round_trip_without_secrets(tmp_path):
    cache = APICache(tmp_path)
    request = LLMRequest(
        prompt="hello",
        prompt_version="v1",
        candidate_item_ids=["i1"],
        provider="deepseek",
        model="deepseek-v4-flash",
    )
    response = LLMResponse(
        raw_output='{"ranked_item_ids":["i1"]}',
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
        metadata={"authorization": "Bearer secret", "response_id": "r1"},
    )

    path = cache.set(request, response)
    assert "secret" not in path.read_text(encoding="utf-8")

    cached = cache.get(request)
    assert cached is not None
    assert cached.cache_hit is True
    assert cached.total_tokens == 7
    assert "authorization" not in cached.metadata
