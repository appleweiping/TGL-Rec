from pathlib import Path

from llm4rec.io.artifacts import write_jsonl
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
