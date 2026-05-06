"""Run local LoRA adapter reranking over frozen candidate artifacts."""

from __future__ import annotations

import time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from llm4rec.evaluation.lora_evaluator import evaluate_lora_predictions
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, iter_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.llm.hf_local_provider import HFLocalProvider, HFLocalProviderConfig
from llm4rec.rankers.local_lora_reranker import LocalLoRARerankExample, LocalLoRAReranker


def run_lora_rerank_eval(
    config_path: str | Path,
    *,
    base_model_path: str | Path,
    limit: int | None = None,
    split: str = "test",
    top_m: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate predictions for configured adapters and evaluate them."""

    config = load_yaml_config(config_path)
    eval_config = dict(config["evaluation_run"])
    run_dir = ensure_dir(resolve_path(eval_config["output_dir"]))
    configured_top_m = int(eval_config.get("top_m_candidates_for_local_lora", 50))
    candidate_limit = int(top_m or configured_top_m)
    adapters = _adapter_specs(eval_config)
    datasets = [str(value) for value in eval_config.get("datasets", [])]
    if not datasets:
        raise ValueError("evaluation_run.datasets must be non-empty")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for adapter in adapters:
        adapter_path = resolve_path(adapter["adapter_path"])
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing adapter path: {adapter_path}")
        provider = HFLocalProvider(
            HFLocalProviderConfig(
                base_model_path=str(base_model_path),
                adapter_path=str(adapter_path),
                tokenizer_path=str(adapter_path.parent / "tokenizer"),
                load_in_4bit=True,
            ),
            dry_run=dry_run,
        )
        reranker = LocalLoRAReranker(
            provider=provider,
            model=str(base_model_path),
            variant=adapter["variant"],
        )
        for dataset in datasets:
            dataset_rows = _build_examples(
                dataset=dataset,
                split=str(split),
                candidate_limit=candidate_limit,
                limit=None if limit is None else int(limit),
            )
            rows.extend(_rank_dataset(reranker, adapter, dataset, dataset_rows))

    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    metrics = evaluate_lora_predictions(predictions_path, run_dir / "metrics")
    metric_rows = _metrics_by_method(rows)
    write_csv_rows(run_dir / "metrics" / "metrics_by_method.csv", metric_rows)
    manifest = {
        "base_model_path": str(base_model_path),
        "candidate_limit": candidate_limit,
        "candidate_selection": "stable_hash_sampled_with_target",
        "dry_run": dry_run,
        "limit_per_dataset_adapter": limit,
        "num_predictions": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "split": split,
        "status": "succeeded",
    }
    write_json(run_dir / "rerank_eval_manifest.json", manifest)
    return {"manifest": manifest, "metrics": metrics, "predictions_path": str(predictions_path)}


def _adapter_specs(eval_config: dict[str, Any]) -> list[dict[str, str]]:
    specs = []
    for raw_path in eval_config.get("adapter_paths", []):
        path = str(raw_path)
        variant = Path(path).parent.name if Path(path).name == "adapter" else Path(path).name
        specs.append({"adapter_path": path, "variant": variant, "method": f"local_8b_lora::{variant}"})
    if not specs:
        raise ValueError("evaluation_run.adapter_paths must be non-empty")
    return specs


def _build_examples(
    *,
    dataset: str,
    split: str,
    candidate_limit: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    artifact_dir = resolve_path(f"outputs/artifacts/protocol_v1/{dataset}")
    split_path = artifact_dir / "splits.jsonl"
    candidate_path = artifact_dir / "candidates.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split artifact: {split_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Missing candidate artifact: {candidate_path}")
    histories = _histories_before_targets(split_path)
    examples = []
    for row in iter_jsonl(candidate_path):
        if str(row.get("split", "")) != split:
            continue
        target = str(row["target_item"])
        candidates = _candidate_items(row, artifact_dir)
        if target not in candidates:
            raise ValueError(f"target missing from candidates: dataset={dataset} target={target}")
        limited = _limit_candidates(
            candidates,
            target=target,
            limit=candidate_limit,
            seed_key=f"{dataset}|{split}|{row['user_id']}|{target}",
        )
        examples.append(
            {
                "candidate_items": limited,
                "domain": row.get("domain"),
                "history": histories.get(str(row["user_id"]), []),
                "target_item": target,
                "user_id": str(row["user_id"]),
            }
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _histories_before_targets(split_path: Path) -> dict[str, list[str]]:
    histories: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in iter_jsonl(split_path):
        if str(row.get("split", "")) != "train":
            continue
        timestamp = float(row.get("timestamp") or 0.0)
        histories[str(row["user_id"])].append((timestamp, str(row["item_id"])))
    return {
        user_id: [item for _, item in sorted(values, key=lambda value: (value[0], value[1]))][-20:]
        for user_id, values in histories.items()
    }


def _candidate_items(row: dict[str, Any], artifact_dir: Path) -> list[str]:
    if isinstance(row.get("candidate_items"), list) and row["candidate_items"]:
        return [str(item) for item in row["candidate_items"]]
    pool_path = artifact_dir / "candidate_pool.json"
    if not pool_path.is_file():
        raise FileNotFoundError(f"Missing shared candidate pool: {pool_path}")
    import json

    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    pool = [str(item) for item in payload.get("candidate_items", [])]
    target = str(row["target_item"])
    if target in pool:
        return pool
    negatives = [str(item) for item in payload.get("negative_pool_for_targets_outside_pool", pool[:-1])]
    return [*negatives, target]


def _limit_candidates(
    candidates: list[str],
    *,
    target: str,
    limit: int,
    seed_key: str | None = None,
) -> list[str]:
    """Create a deterministic sampled candidate subset while preserving the target."""

    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    seed = str(seed_key or target)
    negatives = [item for item in candidates if item != target]
    sampled = sorted(negatives, key=lambda item: _stable_sample_key(seed, item))[: max(limit - 1, 0)]
    insert_at = _stable_position(seed, limit)
    selected = list(sampled)
    selected.insert(insert_at, target)
    return selected[:limit]


def _stable_sample_key(seed_key: str, item: str) -> str:
    return sha256(f"{seed_key}|{item}".encode("utf-8")).hexdigest()


def _stable_position(seed_key: str, limit: int) -> int:
    digest = sha256(f"{seed_key}|target_position".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(limit, 1)


def _rank_dataset(
    reranker: LocalLoRAReranker,
    adapter: dict[str, str],
    dataset: str,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        result = reranker.rank(
            LocalLoRARerankExample(
                user_id=example["user_id"],
                history=example["history"],
                target_item=example["target_item"],
                candidate_items=example["candidate_items"],
                method=adapter["method"],
                metadata={"dataset": dataset},
            )
        )
        rows.append(
            {
                "candidate_items": example["candidate_items"],
                "dataset": dataset,
                "domain": example["domain"],
                "metadata": result.get("metadata", {}),
                "method": adapter["method"],
                "predicted_items": result["predicted_items"],
                "raw_output": result.get("raw_output"),
                "scores": result.get("scores", []),
                "target_item": example["target_item"],
                "user_id": example["user_id"],
            }
        )
    return rows


def _metrics_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "")), str(row.get("method", "")))].append(row)
    output = []
    for (dataset, method), subset in sorted(grouped.items()):
        metrics = evaluate_lora_predictions_to_dict(subset)
        output.append({"dataset": dataset, "method": method, **metrics})
    return output


def evaluate_lora_predictions_to_dict(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate rows without writing side effects for per-method CSV."""

    from llm4rec.metrics.ranking import aggregate_ranking_metrics

    invalid = 0
    total = 0
    parse_success = 0
    for row in rows:
        candidates = {str(item) for item in row.get("candidate_items", [])}
        if row.get("metadata", {}).get("parse_success", False):
            parse_success += 1
        for item in row.get("predicted_items", []):
            total += 1
            if str(item) not in candidates:
                invalid += 1
    return {
        **aggregate_ranking_metrics(rows, ks=(1, 5, 10)),
        "candidate_adherence_rate": 1.0 - invalid / float(total or 1),
        "hallucination_rate": invalid / float(total or 1),
        "parse_success_rate": parse_success / float(len(rows) or 1),
        "validity_rate": 1.0 - invalid / float(total or 1),
    }
