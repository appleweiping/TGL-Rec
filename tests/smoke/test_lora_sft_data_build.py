from pathlib import Path

from llm4rec.io.artifacts import write_jsonl
from llm4rec.trainers.sft_dataset import build_lora_sft_data


def test_lora_sft_data_materialize_outputs_manifest(tmp_path: Path):
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
  max_train_examples: 10
candidate_policy:
  candidate_size_train: 2
  seed: 1
""",
        encoding="utf-8",
    )

    result = build_lora_sft_data(config, materialize=True)[0]

    assert (result.output_dir / "train.jsonl").is_file()
    assert (result.output_dir / "sft_data_manifest.json").is_file()
