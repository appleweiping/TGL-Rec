from pathlib import Path

import pytest

from llm4rec.io.artifacts import read_jsonl, write_json, write_jsonl
from llm4rec.trainers.sft_merge import merge_lora_sft_data


def _write_domain_sft(root: Path, dataset: str, variant: str, row_id: str) -> None:
    domain_dir = root / dataset / variant
    train = {
        "id": f"{dataset}:{row_id}:train",
        "dataset": dataset,
        "messages": [{"role": "assistant", "content": "x"}],
        "metadata": {},
        "variant": variant,
    }
    valid = {
        "id": f"{dataset}:{row_id}:valid",
        "dataset": dataset,
        "messages": [{"role": "assistant", "content": "y"}],
        "metadata": {},
        "variant": variant,
    }
    write_jsonl(domain_dir / "train.jsonl", [train])
    write_jsonl(domain_dir / "valid.jsonl", [valid])
    write_json(domain_dir / "sft_data_manifest.json", {"dataset": dataset, "variant": variant})


def test_merge_lora_sft_data_preserves_domain_order_and_writes_manifest(tmp_path: Path):
    root = tmp_path / "sft"
    variant = "history_only_sft"
    _write_domain_sft(root, "beauty", variant, "u1")
    _write_domain_sft(root, "books", variant, "u2")

    result = merge_lora_sft_data(
        input_root=root,
        output_dir=tmp_path / "merged" / variant,
        variant=variant,
        datasets=["beauty", "books"],
        protocol_version="protocol_week8_large10000_same_candidate",
    )

    rows = read_jsonl(result.output_dir / "train.jsonl")
    assert [row["dataset"] for row in rows] == ["beauty", "books"]
    assert rows[0]["metadata"]["merged_domain"] == "beauty"
    assert result.manifest["num_train_rows"] == 2
    assert (result.output_dir / "sft_merge_manifest.json").is_file()


def test_merge_lora_sft_data_rejects_duplicate_ids(tmp_path: Path):
    root = tmp_path / "sft"
    variant = "history_only_sft"
    _write_domain_sft(root, "beauty", variant, "same")
    _write_domain_sft(root, "books", variant, "same")
    duplicate = read_jsonl(root / "books" / variant / "train.jsonl")[0]
    duplicate["id"] = "beauty:same:train"
    write_jsonl(root / "books" / variant / "train.jsonl", [duplicate])

    with pytest.raises(ValueError, match="Duplicate SFT row ids"):
        merge_lora_sft_data(
            input_root=root,
            output_dir=tmp_path / "merged" / variant,
            variant=variant,
            datasets=["beauty", "books"],
            protocol_version="protocol_week8_large10000_same_candidate",
        )
