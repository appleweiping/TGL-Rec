from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.response_cache import ResponseCache


def test_response_cache_round_trip(tmp_path):
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    request = LLMRequest(
        prompt="hello",
        prompt_version="v1",
        candidate_item_ids=["i1"],
        provider="mock",
        model="mock",
        decoding_params={"temperature": 0.0},
        metadata={"dataset_run_id": "run"},
    )
    response = LLMResponse(raw_output='{"ranked_item_ids":["i1"]}', provider="mock", model="mock")
    cache.set(request, response)
    cached = cache.get(request)

    assert cached is not None
    assert cached.cache_hit
    assert cached.raw_output == response.raw_output
    assert cache.path_for(request).is_file()

