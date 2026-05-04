from llm4rec.llm.base import LLMRequest
from llm4rec.llm.hf_local_provider import HFLocalProvider, HFLocalProviderConfig


def test_hf_local_provider_dry_run_returns_candidate_json():
    provider = HFLocalProvider(HFLocalProviderConfig(base_model_path="missing"), dry_run=True)
    response = provider.generate(
        LLMRequest(
            prompt="rank",
            prompt_version="history_only_sft",
            candidate_item_ids=["i1", "i2"],
            provider="hf_local",
            model="missing",
        )
    )

    assert "i1" in response.raw_output
    assert response.metadata["dry_run"] is True
