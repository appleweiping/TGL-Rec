from llm4rec.llm.hf_local_provider import HFLocalProvider, HFLocalProviderConfig
from llm4rec.rankers.local_lora_reranker import LocalLoRARerankExample, LocalLoRAReranker


def test_local_lora_reranker_dry_run_ranks_candidates():
    provider = HFLocalProvider(HFLocalProviderConfig(base_model_path="missing"), dry_run=True)
    reranker = LocalLoRAReranker(provider=provider, model="missing", variant="history_only_sft")

    result = reranker.rank(
        LocalLoRARerankExample(
            user_id="u1",
            history=["i0"],
            target_item="i1",
            candidate_items=["i1", "i2"],
        )
    )

    assert result["predicted_items"][0] == "i1"
    assert result["metadata"]["parse_success"] is True
