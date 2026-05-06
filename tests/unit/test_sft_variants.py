from llm4rec.trainers.sft_variants import get_sft_variant, sft_variant_names


def test_sft_variant_registry_includes_reference_baselines():
    names = sft_variant_names()

    assert "history_only_sft" in names
    assert "temporal_evidence_sft" in names
    assert "reference_preference_sft" in names
    assert get_sft_variant("reference_preference_sft").family == "reference_baseline"
    assert "Preference evidence" in get_sft_variant("reference_preference_sft").evidence_block()
