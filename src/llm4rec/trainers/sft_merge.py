"""Merge per-domain LoRA SFT artifacts without resampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, iter_jsonl, sha256_file, write_json, write_jsonl


@dataclass(frozen=True)
class SFTMergeResult:
    """Summary for a merged SFT artifact."""

    output_dir: Path
    manifest: dict[str, Any]


def merge_lora_sft_data(
    *,
    input_root: str | Path,
    output_dir: str | Path,
    variant: str,
    datasets: list[str],
    protocol_version: str,
    dry_run: bool = False,
) -> SFTMergeResult:
    """Merge already materialized domain SFT files into one training directory.

    This function deliberately does not sample, shuffle, or regenerate examples.
    It preserves per-domain row order from the materialized artifacts and records
    every input checksum so the server run can be audited later.
    """

    if not datasets:
        raise ValueError("datasets must be non-empty")
    root = resolve_path(input_root)
    output = resolve_path(output_dir)
    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    inputs = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for dataset in datasets:
        dataset_name = str(dataset)
        domain_dir = root / dataset_name / variant
        train_path = domain_dir / "train.jsonl"
        valid_path = domain_dir / "valid.jsonl"
        manifest_path = domain_dir / "sft_data_manifest.json"
        if not train_path.is_file():
            raise FileNotFoundError(f"Missing domain train SFT file: {train_path}")
        if not valid_path.is_file():
            raise FileNotFoundError(f"Missing domain valid SFT file: {valid_path}")
        domain_train = [_with_merge_metadata(row, dataset_name, "train") for row in iter_jsonl(train_path)]
        domain_valid = [_with_merge_metadata(row, dataset_name, "valid") for row in iter_jsonl(valid_path)]
        for row in [*domain_train, *domain_valid]:
            row_id = str(row.get("id", ""))
            if row_id in seen_ids:
                duplicate_ids.append(row_id)
            seen_ids.add(row_id)
        train_rows.extend(domain_train)
        valid_rows.extend(domain_valid)
        inputs.append(
            {
                "dataset": dataset_name,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
                "train_path": str(train_path),
                "train_rows": len(domain_train),
                "train_sha256": sha256_file(train_path),
                "valid_path": str(valid_path),
                "valid_rows": len(domain_valid),
                "valid_sha256": sha256_file(valid_path),
            }
        )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate SFT row ids across merged inputs: {preview}")

    manifest = {
        "datasets": [str(dataset) for dataset in datasets],
        "dry_run": dry_run,
        "input_root": str(root),
        "inputs": inputs,
        "merge_policy": "concatenate_domain_artifacts_preserve_domain_order_no_resampling",
        "num_train_rows": len(train_rows),
        "num_valid_rows": len(valid_rows),
        "output_dir": str(output),
        "protocol_version": protocol_version,
        "variant": variant,
    }
    if not dry_run:
        ensure_dir(output)
        write_jsonl(output / "train.jsonl", train_rows)
        write_jsonl(output / "valid.jsonl", valid_rows)
        write_json(output / "sft_merge_manifest.json", manifest)
    return SFTMergeResult(output_dir=output, manifest=manifest)


def _with_merge_metadata(row: dict[str, Any], dataset: str, split: str) -> dict[str, Any]:
    output = dict(row)
    output["dataset"] = str(output.get("dataset", dataset))
    metadata = dict(output.get("metadata", {}))
    metadata["merged_domain"] = dataset
    metadata["merged_sft_split"] = split
    output["metadata"] = metadata
    return output
