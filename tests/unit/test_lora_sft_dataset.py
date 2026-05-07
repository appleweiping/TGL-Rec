from pathlib import Path

from llm4rec.io.artifacts import write_jsonl
from llm4rec.io.artifacts import read_jsonl
from llm4rec.trainers.sft_dataset import build_lora_sft_data


def test_lora_sft_dataset_uses_train_only_targets(tmp_path: Path):
    split = tmp_path / "splits.jsonl"
    write_jsonl(
        split,
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "split": "train"},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "split": "train"},
            {"user_id": "u1", "item_id": "i3", "timestamp": 3, "split": "test"},
        ],
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"""
dataset_artifacts:
  tiny:
    split_artifact: {split}
sft:
  variant: history_only_sft
  output_root: {tmp_path / "out"}
  dry_run_output_dir: {tmp_path / "dry"}
  max_train_examples: 10
candidate_policy:
  candidate_size_train: 2
  seed: 1
""",
        encoding="utf-8",
    )

    result = build_lora_sft_data(config, dry_run=True)[0]

    assert result.leakage_audit["leakage_free"] is True
    assert result.manifest["num_train_rows"] >= 1


def test_lora_sft_dataset_accepts_reference_variant(tmp_path: Path):
    split = tmp_path / "splits.jsonl"
    write_jsonl(
        split,
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "split": "train"},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "split": "train"},
        ],
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"""
dataset_artifacts:
  tiny:
    split_artifact: {split}
sft:
  variant: reference_preference_sft
  output_root: {tmp_path / "out"}
  dry_run_output_dir: {tmp_path / "dry"}
  max_train_examples: 10
candidate_policy:
  candidate_size_train: 2
  seed: 1
""",
        encoding="utf-8",
    )

    result = build_lora_sft_data(config, dry_run=True)[0]

    assert result.manifest["variant"] == "reference_preference_sft"
    assert result.leakage_audit["leakage_free"] is True


def test_lora_sft_dataset_persists_configured_protocol_version(tmp_path: Path):
    split = tmp_path / "splits.jsonl"
    write_jsonl(
        split,
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "split": "train"},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "split": "train"},
        ],
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"""
protocol_version: protocol_week8_large10000_same_candidate
dataset_artifacts:
  books:
    split_artifact: {split}
sft:
  variant: history_only_sft
  output_root: {tmp_path / "out"}
  dry_run_output_dir: {tmp_path / "dry"}
  max_train_examples: 10
candidate_policy:
  candidate_size_train: 2
  seed: 1
""",
        encoding="utf-8",
    )

    result = build_lora_sft_data(config, dry_run=True)[0]

    assert result.manifest["protocol_version"] == "protocol_week8_large10000_same_candidate"


def test_temporal_evidence_sft_includes_candidate_specific_evidence(tmp_path: Path):
    split = tmp_path / "splits.jsonl"
    write_jsonl(
        split,
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "split": "train"},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "split": "train"},
        ],
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"""
dataset_artifacts:
  tiny:
    split_artifact: {split}
sft:
  variant: temporal_evidence_sft
  output_root: {tmp_path / "out"}
  dry_run_output_dir: {tmp_path / "dry"}
  max_train_examples: 10
candidate_policy:
  candidate_size_train: 2
  seed: 1
""",
        encoding="utf-8",
    )

    result = build_lora_sft_data(config, materialize=True)[0]
    rows = read_jsonl(result.output_dir / "train.jsonl")

    assert "Candidate temporal evidence JSON" in rows[0]["messages"][1]["content"]
    assert rows[0]["metadata"]["candidate_specific_evidence"] is True
