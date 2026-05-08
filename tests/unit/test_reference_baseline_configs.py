from __future__ import annotations

from pathlib import Path

import yaml

from llm4rec.baselines.reference_methods import get_reference_method


ROOT = Path(__file__).resolve().parents[2]


def _load_config(name: str) -> dict:
    path = ROOT / "configs" / "experiments" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_reference_training_configs_declare_official_contract() -> None:
    configs = {
        "paper_lora_8b_reference_collaborative.yaml": "cllm4rec_qwen_lora",
        "paper_lora_8b_reference_long_tail.yaml": "llm_esr_qwen_lora",
        "paper_lora_8b_reference_preference.yaml": "review_pref_reasoning_qwen_lora",
        "paper_lora_8b_reference_semantic.yaml": "rlmrec_qwen_lora",
    }

    for config_name, baseline_id in configs.items():
        config = _load_config(config_name)
        contract = config["baseline_contract"]
        official_training = config["official_training"]
        spec = get_reference_method(baseline_id)

        assert contract["reference_method_id"] == baseline_id
        assert contract["base_model_policy"] == "unified_qwen3_8b_base_model"
        assert contract["comparison_protocol"] == "official_default_qwen3_8b_lora"
        assert contract["adapter_training_policy"] == "adapt_official_algorithm_to_project_lora_regime"
        assert contract["baseline_hyperparameter_policy"] == (
            "official_default_or_paper_recommended_hyperparameters"
        )
        assert contract["ours_hyperparameter_policy"] == "validation_tuning_with_logged_search_space"
        assert contract["scoring_policy"] == "preserve_each_official_baseline_scoring_logic"
        assert contract["scaffold_only"] is True
        assert official_training["policy"] == "blocked_until_official_code_adapter"
        assert official_training["target_policy_after_promotion"] == "preserve_official_algorithm"
        assert official_training["lora_or_adapter_required"] == "project_lora_or_qlora_regime"
        assert official_training["hyperparameter_policy"] == (
            "official_default_or_paper_recommended_hyperparameters"
        )
        assert official_training["scaffold_only"] is True
        assert config["sft"]["scaffold_only"] is True
        assert "not_official_baseline" in config["sft"]["container_policy"]
        assert "training" not in config
        assert spec.official_code_status == "official_code_identified"
        assert spec.reportable_baseline is False


def test_reference_eval_config_tracks_variant_method_mapping() -> None:
    config = _load_config("paper_lora_8b_reference_rerank_eval.yaml")
    contract = config["baseline_contract"]

    assert sorted(contract["intended_reference_method_ids"]) == sorted(
        [
            "cllm4rec_qwen_lora",
            "llm_esr_qwen_lora",
            "review_pref_reasoning_qwen_lora",
            "rlmrec_qwen_lora",
        ]
    )
    assert contract["variant_reference_method_ids"]["reference_collaborative_sft"] == (
        "cllm4rec_qwen_lora"
    )
    assert contract["scaffold_only"] is True
    assert contract["comparison_protocol"] == "official_default_qwen3_8b_lora"
    assert "method_specific_scoring_export" in contract["promotion_required_before_main_table"]
    assert "checkpoint_or_adapter_paths" in config["evaluation_run"]
