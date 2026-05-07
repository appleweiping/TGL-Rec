from llm4rec.evaluation.lora_rerank import _adapter_specs, _limit_candidates


def test_limit_candidates_samples_negatives_and_preserves_target_position():
    candidates = [f"i{index}" for index in range(100)]
    target = "i99"

    limited = _limit_candidates(candidates, target=target, limit=10, seed_key="u1|i99")
    repeated = _limit_candidates(candidates, target=target, limit=10, seed_key="u1|i99")

    assert limited == repeated
    assert len(limited) == 10
    assert target in limited
    assert limited != candidates[:9] + [target]
    assert limited[-1] != target


def test_adapter_specs_attach_reference_baseline_provenance():
    adapters = _adapter_specs(
        {
            "checkpoint_or_adapter_paths": [
                "outputs/paper_runs/protocol_v1/lora_8b/reference_collaborative_sft/adapter"
            ],
            "do_not_merge_into_main_accuracy_table": True,
        },
        baseline_contract={
            "status": "candidate_scaffold_eval_not_reportable",
            "variant_reference_method_ids": {
                "reference_collaborative_sft": "cllm4rec_qwen_lora",
            },
        },
    )

    provenance = adapters[0]["baseline_provenance"]
    assert adapters[0]["reference_method_id"] == "cllm4rec_qwen_lora"
    assert provenance["baseline_id"] == "cllm4rec_qwen_lora"
    assert provenance["official_code_status"] == "official_code_identified"
    assert provenance["reportable_baseline"] is False
    assert provenance["do_not_merge_into_main_accuracy_table"] is True
