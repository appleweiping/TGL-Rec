from __future__ import annotations

import pytest

from llm4rec.baselines.reference_methods import (
    ReferenceBaselineNotImplementedError,
    get_reference_method,
    reference_method_names,
    require_implemented_reference_method,
)


def test_reference_method_registry_selects_concrete_methods() -> None:
    names = reference_method_names()

    # Historical scaffold registry only. The active main baselines now come from
    # configs/baselines/pony_official_external.yaml.
    assert "slmrec_distill_qwen_lora" in names
    assert "llm_esr_qwen_lora" in names
    assert "controllable_rec_qwen_lora" in names
    assert "cllm4rec_qwen_lora" in names
    assert "rlmrec_qwen_lora" in names
    assert "transrec_qwen_lora" in names
    assert "review_pref_reasoning_qwen_lora" in names
    assert len(names) == 7


def test_reference_methods_are_not_reportable_until_implemented() -> None:
    spec = get_reference_method("cllm4rec_qwen_lora")

    # These local reference specs are preserved for provenance, not for the
    # active Phase 10 main baseline table.
    assert spec.reportable_baseline is False
    assert spec.implementation_status == "not_implemented"
    assert spec.official_code_status == "official_code_identified"
    assert spec.official_code_url == "https://github.com/yaochenzhu/LLM4Rec"
    assert spec.base_model_policy == "unified_qwen3_8b_base_model"
    assert spec.comparison_protocol == "official_default_qwen3_8b_lora"
    assert spec.adapter_training_policy == "preserve_official_id_token_prompt_head_and_regularization_algorithm"
    assert spec.finetune_regime == "project_lora_or_qlora_regime"
    assert spec.hyperparameter_policy == (
        "official_default_hyperparameters_for_baselines_validation_tuning_for_ours"
    )
    assert spec.scoring_policy == "preserve_official_item_prediction_head_or_candidate_scoring_logic"
    assert spec.protocol_controls == ("data", "candidate_sets", "splits", "metrics", "prediction_schema")
    assert "item prediction head" in spec.preserve_components
    with pytest.raises(ReferenceBaselineNotImplementedError):
        require_implemented_reference_method("cllm4rec_qwen_lora")


def test_reference_method_metadata_is_serializable_provenance() -> None:
    spec = get_reference_method("llm_esr_qwen_lora")

    metadata = spec.to_metadata()

    assert metadata["baseline_id"] == "llm_esr_qwen_lora"
    assert metadata["official_code_status"] == "official_code_identified"
    assert metadata["implementation_status"] == "not_implemented"
    assert metadata["reportable_baseline"] is False
    assert metadata["base_model_policy"] == "unified_qwen3_8b_base_model"
    assert metadata["comparison_protocol"] == "official_default_qwen3_8b_lora"
    assert metadata["adapter_training_policy"] == (
        "preserve_official_long_tail_dual_view_and_self_distillation_algorithm"
    )
    assert metadata["finetune_regime"] == "project_lora_or_qlora_regime"
    assert metadata["hyperparameter_policy"] == (
        "official_default_hyperparameters_for_baselines_validation_tuning_for_ours"
    )
    assert metadata["scoring_policy"] == "preserve_official_long_tail_sequential_scoring_when_feasible"
    assert metadata["protocol_controls"] == [
        "data",
        "candidate_sets",
        "splits",
        "metrics",
        "prediction_schema",
    ]


def test_reference_method_guard_rejects_missing_official_code() -> None:
    spec = get_reference_method("controllable_rec_qwen_lora")

    assert spec.official_code_status == "no_official_code_identified"
    assert spec.official_code_url is None
    with pytest.raises(ReferenceBaselineNotImplementedError):
        require_implemented_reference_method("controllable_rec_qwen_lora")
