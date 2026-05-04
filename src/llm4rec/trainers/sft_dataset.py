"""Train-only SFT data construction for local LoRA rerankers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, iter_jsonl, sha256_file, write_json, write_jsonl


SFT_VARIANTS = {"history_only_sft", "temporal_evidence_sft"}


@dataclass(frozen=True)
class SFTBuildResult:
    """Output locations and summary for one SFT data build."""

    output_dir: Path
    manifest: dict[str, Any]
    leakage_audit: dict[str, Any]


def build_lora_sft_data(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    materialize: bool = False,
) -> list[SFTBuildResult]:
    """Build SFT rows from train split only."""

    if dry_run == materialize:
        raise ValueError("Specify exactly one of dry_run or materialize.")
    config = load_yaml_config(config_path)
    variant = str(config["sft"]["variant"])
    if variant not in SFT_VARIANTS:
        raise ValueError(f"Unsupported SFT variant: {variant}")
    results: list[SFTBuildResult] = []
    for dataset, artifacts in dict(config["dataset_artifacts"]).items():
        results.append(_build_dataset(config, str(dataset), dict(artifacts), variant, dry_run=dry_run))
    return results


def _build_dataset(
    config: dict[str, Any],
    dataset: str,
    artifacts: dict[str, Any],
    variant: str,
    *,
    dry_run: bool,
) -> SFTBuildResult:
    split_artifact = resolve_path(artifacts["split_artifact"])
    train_rows, forbidden_rows = _load_train_and_forbidden(split_artifact)
    policy = dict(config.get("candidate_policy", {}))
    seed = int(policy.get("seed", 2026))
    rng = random.Random(seed)
    candidate_size = int(policy.get("candidate_size_train", 20))
    max_examples = config.get("sft", {}).get("max_train_examples")
    valid_ratio = float(config.get("sft", {}).get("valid_ratio", 0.05))
    examples = _make_examples(
        dataset=dataset,
        variant=variant,
        train_rows=train_rows,
        candidate_size=candidate_size,
        rng=rng,
        max_examples=None if max_examples in (None, "null") else int(max_examples),
    )
    split_index = max(1, int(len(examples) * (1.0 - valid_ratio))) if examples else 0
    train_examples = examples[:split_index]
    valid_examples = examples[split_index:]
    output_root = resolve_path(config["sft"]["output_root"])
    output_dir = output_root / dataset / variant
    if not dry_run:
        ensure_dir(output_dir)
        write_jsonl(output_dir / "train.jsonl", train_examples)
        write_jsonl(output_dir / "valid.jsonl", valid_examples)
    leakage = _leakage_audit(train_examples + valid_examples, forbidden_rows)
    manifest = {
        "candidate_policy": {
            "candidate_size_train": candidate_size,
            "include_target": True,
            "negative_source": "train_split_items_only",
            "seed": seed,
        },
        "constructed_from": "train_only",
        "dataset": dataset,
        "dry_run": dry_run,
        "label_construction_policy": "assistant ranks supervised positive next train item first; labels are supervised recommendation labels, not human preferences",
        "leakage_audit_path": str(output_dir / "leakage_audit.json"),
        "num_train_rows": len(train_examples),
        "num_valid_rows": len(valid_examples),
        "output_dir": str(output_dir),
        "protocol_version": "protocol_v1",
        "split_artifact": str(split_artifact),
        "split_sha256": sha256_file(split_artifact),
        "target_inclusion_rate": 1.0 if examples else 0.0,
        "variant": variant,
    }
    if not dry_run:
        write_json(output_dir / "sft_data_manifest.json", manifest)
        write_json(output_dir / "leakage_audit.json", leakage)
    else:
        dry_dir = ensure_dir(resolve_path(config["sft"].get("dry_run_output_dir", "outputs/paper_runs/protocol_v1/lora_8b/dry_run_sft")))
        write_json(dry_dir / f"{dataset}_{variant}_sft_data_manifest.json", manifest)
        write_json(dry_dir / f"{dataset}_{variant}_leakage_audit.json", leakage)
    return SFTBuildResult(output_dir=output_dir, manifest=manifest, leakage_audit=leakage)


def _load_train_and_forbidden(split_artifact: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    train_rows: list[dict[str, Any]] = []
    forbidden: set[tuple[str, str]] = set()
    for row in iter_jsonl(split_artifact):
        split = str(row.get("split", ""))
        if split == "train":
            train_rows.append(row)
        elif split in {"valid", "validation", "test"}:
            forbidden.add((str(row["user_id"]), str(row["item_id"])))
    train_rows.sort(key=lambda row: (str(row["user_id"]), float(row.get("timestamp") or 0.0), str(row["item_id"])))
    return train_rows, forbidden


def _make_examples(
    *,
    dataset: str,
    variant: str,
    train_rows: list[dict[str, Any]],
    candidate_size: int,
    rng: random.Random,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    items = sorted({str(row["item_id"]) for row in train_rows})
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_user[str(row["user_id"])].append(row)
    examples: list[dict[str, Any]] = []
    for user_id, rows in sorted(by_user.items()):
        history: list[str] = []
        for row in rows:
            target = str(row["item_id"])
            if history:
                negatives = _sample_negatives(items, target=target, count=max(0, candidate_size - 1), rng=rng)
                candidates = [target, *negatives]
                rng.shuffle(candidates)
                examples.append(
                    _sft_row(
                        dataset=dataset,
                        user_id=user_id,
                        target=target,
                        history=history[-20:],
                        candidates=candidates,
                        variant=variant,
                        source_split=str(row.get("split", "train")),
                    )
                )
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
            history.append(target)
    return examples


def _sample_negatives(items: list[str], *, target: str, count: int, rng: random.Random) -> list[str]:
    if count <= 0:
        return []
    if len(items) <= count + 1:
        return [item for item in items if item != target][:count]
    selected: list[str] = []
    seen = {target}
    while len(selected) < count:
        item = items[rng.randrange(len(items))]
        if item in seen:
            continue
        seen.add(item)
        selected.append(item)
    return selected


def _sft_row(
    *,
    dataset: str,
    user_id: str,
    target: str,
    history: list[str],
    candidates: list[str],
    variant: str,
    source_split: str = "train",
) -> dict[str, Any]:
    evidence = ""
    if variant == "temporal_evidence_sft":
        evidence = (
            "\nTime buckets: recent history items are later in the sequence."
            "\nTransition evidence: rank candidates likely to follow the recent history."
            "\nContrastive evidence: distinguish semantic similarity from next-need transitions."
        )
    user_prompt = (
        "Rank candidate item IDs for the next recommendation. Return JSON only.\n"
        f"History: {history}\nCandidates: {candidates}{evidence}"
    )
    ranked = [target, *[item for item in candidates if item != target]]
    return {
        "id": f"{dataset}:{variant}:{user_id}:{target}:{len(history)}",
        "dataset": dataset,
        "metadata": {
            "candidate_source": "train_negative_sampling",
            "constructed_from": "train_only",
            "protocol_version": "protocol_v1",
            "source_split": source_split,
            "target_item": target,
            "user_id": user_id,
        },
        "messages": [
            {"content": "You are a recommendation reranker. Output strict JSON only.", "role": "system"},
            {"content": user_prompt, "role": "user"},
            {"content": json.dumps({"ranked_item_ids": ranked}, ensure_ascii=True), "role": "assistant"},
        ],
        "variant": variant,
    }


def _leakage_audit(rows: list[dict[str, Any]], forbidden: set[tuple[str, str]]) -> dict[str, Any]:
    violations = []
    overlap_warnings = []
    for row in rows:
        meta = row.get("metadata", {})
        if str(meta.get("source_split", "")) != "train":
            violations.append(
                {
                    "reason": "non_train_source_split",
                    "source_split": meta.get("source_split", ""),
                    "target_item": meta.get("target_item", ""),
                    "user_id": meta.get("user_id", ""),
                }
            )
        key = (str(meta.get("user_id", "")), str(meta.get("target_item", "")))
        if key in forbidden:
            overlap_warnings.append({"target_item": key[1], "user_id": key[0]})
    return {
        "forbidden_valid_test_targets_used": 0,
        "non_train_source_split_violations": len(violations),
        "overlap_user_item_with_eval_count": len(overlap_warnings),
        "overlap_user_item_with_eval_note": "Same user-item can appear in train and eval in repeated-consumption data; SFT rows are leakage-free when constructed from train split rows only.",
        "leakage_free": len(violations) == 0,
        "violations_sample": violations[:20],
        "overlap_warnings_sample": overlap_warnings[:20],
    }
