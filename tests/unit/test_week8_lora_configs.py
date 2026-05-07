from pathlib import Path

from llm4rec.experiments.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[2]


def test_week8_lora_training_configs_use_four_domain_sft_dirs():
    configs = [
        ("week8_lora_8b_history_only.yaml", "history_only_sft"),
        ("week8_lora_8b_temporal_evidence.yaml", "temporal_evidence_sft"),
    ]

    for filename, variant in configs:
        config = load_yaml_config(ROOT / "configs" / "experiments" / filename)

        assert config["sft"]["variant"] == variant
        assert config["sft"]["data_dir"].endswith(f"four_domain/{variant}")


def test_week8_lora_eval_preserves_external_candidates_and_is_guarded():
    config = load_yaml_config(ROOT / "configs" / "experiments" / "week8_lora_8b_rerank_eval.yaml")

    eval_config = config["evaluation_run"]
    assert eval_config["candidate_selection"] == "preserve_external_candidates"
    assert eval_config["do_not_merge_into_main_accuracy_table"] is True
    assert eval_config["datasets"] == ["beauty", "books", "electronics", "movies"]
