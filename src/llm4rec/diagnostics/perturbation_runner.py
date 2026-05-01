"""Sequence/time perturbation diagnostics for MovieLens-style data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
from llm4rec.data.sequence_transforms import apply_sequence_transform
from llm4rec.data.time_features import consecutive_time_gaps
from llm4rec.diagnostics.statistics import compare_prediction_sets, compute_metric_deltas
from llm4rec.evaluation.diagnostic_evaluator import evaluate_diagnostic_predictions
from llm4rec.experiments.config import (
    load_yaml_config,
    resolve_experiment_config,
    resolve_path,
    save_resolved_config,
)
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.runner import _make_ranker, _resolve_method_configs
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.rankers.base import RankingExample
from llm4rec.utils.env import collect_environment

DEFAULT_VARIANTS = [
    "original",
    "reversed",
    "shuffled_seed_0",
    "shuffled_seed_1",
    "shuffled_seed_2",
    "recent_5",
    "recent_10",
    "remove_recent_5",
    "popularity_sorted",
    "timestamp_bucketed_prompt_ready",
]


def run_perturbation_experiment(config_path: str | Path) -> Path:
    """Run baseline rankers over shared candidate sets with perturbed histories."""

    config = _resolve_diagnostic_config(config_path)
    run_dir, logger = _prepare_run(config)
    processed_dir = _prepare_processed_dataset(config, run_dir, logger)
    dataset = config["dataset"]
    evaluation = config.get("evaluation", {})
    diagnostic = config.get("diagnostic", {})
    seed = int(config.get("experiment", {}).get("seed", dataset.get("seed", 0)))
    variants = [str(name) for name in diagnostic.get("variants", DEFAULT_VARIANTS)]
    top_k = int(evaluation.get("top_k", 5))
    ks = tuple(int(k) for k in evaluation.get("ks", [1, 3, 5]))
    candidate_protocol = str(evaluation.get("candidate_protocol", dataset.get("candidate_protocol", "full_catalog")))

    all_interactions = read_jsonl(processed_dir / "interactions.jsonl")
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    candidate_rows = read_jsonl(processed_dir / "candidates.jsonl")
    item_catalog = {str(row["item_id"]) for row in item_rows}
    method_configs = _resolve_method_configs(config.get("methods", []))
    if not method_configs:
        raise ValueError("Perturbation diagnostics require at least one baseline method.")

    predictions_dir = ensure_dir(run_dir / "artifacts" / "predictions")
    ensure_dir(run_dir / "artifacts" / "candidates")
    write_jsonl(run_dir / "artifacts" / "candidates" / "candidates.jsonl", candidate_rows)
    examples = [_candidate_to_example(row, all_interactions) for row in candidate_rows if row["split"] == "test"]
    popularity = Counter(str(row["item_id"]) for row in train_rows)
    sequence_artifact = _sequence_artifact(examples, variants, popularity=popularity, seed=seed)
    write_json(run_dir / "diagnostics" / "sequence_perturbation.json", sequence_artifact)

    all_predictions: list[dict[str, Any]] = []
    for method_config in method_configs:
        ranker = _make_ranker(method_config, seed=seed)
        method_name = str(method_config.get("name", getattr(ranker, "name", "unknown")))
        logger.info(f"fitting baseline={method_name}")
        ranker.fit(train_rows, item_rows)
        ranker.save_artifact(run_dir / "artifacts" / "methods" / method_name)
        for variant in variants:
            variant_predictions: list[dict[str, Any]] = []
            for example in examples:
                transformed_history, metadata = _transform_example_history(
                    example,
                    variant=variant,
                    popularity=popularity,
                    seed=seed,
                )
                result = ranker.rank(
                    RankingExample(
                        user_id=example.user_id,
                        history=transformed_history,
                        target_item=example.target_item,
                        candidate_items=example.candidate_items,
                        domain=example.domain,
                        metadata=example.metadata,
                    )
                )
                prediction = {
                    "candidate_items": example.candidate_items,
                    "domain": example.domain,
                    "metadata": {
                        **result.metadata,
                        **metadata,
                        "method_reportable": bool(method_config.get("reportable", True)),
                        "perturbation": variant,
                        "phase": "phase2b",
                    },
                    "method": method_name,
                    "predicted_items": result.items,
                    "raw_output": result.raw_output,
                    "scores": result.scores,
                    "target_item": example.target_item,
                    "user_id": example.user_id,
                }
                variant_predictions.append(prediction)
                all_predictions.append(prediction)
            write_jsonl(predictions_dir / f"{method_name}__{variant}.jsonl", variant_predictions)
    write_jsonl(run_dir / "predictions.jsonl", all_predictions)
    write_jsonl(predictions_dir / "all_predictions.jsonl", all_predictions)

    metric_rows = evaluate_diagnostic_predictions(
        prediction_rows=all_predictions,
        item_catalog=item_catalog,
        ks=ks,
        candidate_protocol=candidate_protocol,
    )
    write_json(run_dir / "metrics.json", {"metrics": metric_rows})
    write_csv_rows(run_dir / "metrics.csv", metric_rows)
    write_csv_rows(run_dir / "perturbation_results.csv", metric_rows)

    deltas = compute_metric_deltas(metric_rows, metrics=("Recall@5", "NDCG@5"))
    write_csv_rows(run_dir / "perturbation_deltas.csv", deltas)
    overlap_rows = _prediction_overlap_rows(all_predictions, variants=variants, k=top_k)
    write_csv_rows(run_dir / "prediction_overlap.csv", overlap_rows)
    logger.info("perturbation diagnostics completed")
    return run_dir


def _resolve_diagnostic_config(config_path: str | Path) -> dict[str, Any]:
    config = resolve_experiment_config(config_path)
    if "methods" in config:
        return config
    return load_yaml_config(config_path)


def _prepare_run(config: dict[str, Any]) -> tuple[Path, RunLogger]:
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", config.get("dataset", {}).get("seed", 0)))
    set_global_seed(seed)
    run_id = str(experiment.get("run_id", "phase2b_movielens_diagnostics"))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = ensure_dir(output_root / run_id)
    ensure_dir(run_dir / "artifacts")
    ensure_dir(run_dir / "diagnostics")
    logger = RunLogger(run_dir / "logs.txt")
    logger.info(f"starting perturbation diagnostics run_id={run_id}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    return run_dir, logger


def _prepare_processed_dataset(config: dict[str, Any], run_dir: Path, logger: RunLogger) -> Path:
    dataset = dict(config["dataset"])
    if bool(dataset.get("preprocess", True)):
        dataset["output_dir"] = str(run_dir / "artifacts" / "processed_dataset")
        logger.info("running MovieLens-style preprocessing")
        return preprocess_from_config({"dataset": dataset}).output_dir
    return resolve_path(dataset["output_dir"])


def _candidate_to_example(
    candidate_row: dict[str, Any],
    all_interactions: list[dict[str, Any]],
) -> RankingExample:
    user_id = str(candidate_row["user_id"])
    target_item = str(candidate_row["target_item"])
    split = str(candidate_row["split"])
    targets = [
        row
        for row in all_interactions
        if str(row["user_id"]) == user_id
        and str(row["item_id"]) == target_item
        and str(row.get("split")) == split
    ]
    if not targets:
        raise ValueError(f"Missing target interaction for user={user_id} item={target_item} split={split}")
    target = sorted(
        targets,
        key=lambda row: float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
    )[-1]
    target_ts = None if target.get("timestamp") is None else float(target["timestamp"])
    history_rows = []
    for row in all_interactions:
        if str(row["user_id"]) != user_id:
            continue
        timestamp = None if row.get("timestamp") is None else float(row["timestamp"])
        if target_ts is not None and timestamp is not None and timestamp >= target_ts:
            continue
        if str(row.get("split")) == "test":
            continue
        history_rows.append(row)
    history_rows = sorted(
        history_rows,
        key=lambda row: (
            float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
            str(row["item_id"]),
        ),
    )
    return RankingExample(
        user_id=user_id,
        history=[str(row["item_id"]) for row in history_rows],
        target_item=target_item,
        candidate_items=[str(item) for item in candidate_row["candidate_items"]],
        domain=None if candidate_row.get("domain") is None else str(candidate_row.get("domain")),
        metadata={"history_rows": history_rows, "split": split},
    )


def _transform_example_history(
    example: RankingExample,
    *,
    variant: str,
    popularity: Counter[str],
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    if variant == "original":
        return list(example.history), {"history_transform": "original"}
    if variant == "reversed":
        return list(reversed(example.history)), {"history_transform": "reversed"}
    if variant.startswith("shuffled_seed_"):
        variant_seed = int(variant.rsplit("_", 1)[-1])
        return (
            apply_sequence_transform(
                example.history,
                transform="shuffled",
                seed=variant_seed,
                popularity=popularity,
            ),
            {"history_transform": "shuffled", "variant_seed": variant_seed},
        )
    if variant.startswith("recent_"):
        k = int(variant.rsplit("_", 1)[-1])
        return (
            apply_sequence_transform(example.history, transform="recent_k", k=k),
            {"history_transform": "recent_k", "k": k},
        )
    if variant.startswith("remove_recent_"):
        k = int(variant.rsplit("_", 1)[-1])
        return (
            apply_sequence_transform(example.history, transform="remove_recent_k", k=k),
            {"history_transform": "remove_recent_k", "k": k},
        )
    if variant == "popularity_sorted":
        return (
            apply_sequence_transform(
                example.history,
                transform="popularity_sorted",
                popularity=popularity,
                seed=seed,
            ),
            {"history_transform": "popularity_sorted"},
        )
    if variant == "timestamp_bucketed_prompt_ready":
        time_tags = consecutive_time_gaps(list(example.metadata.get("history_rows", [])))
        return (
            list(example.history),
            {
                "history_transform": "timestamp_bucketed_prompt_ready",
                "time_tagged_history": [
                    {"gap_bucket": row["gap_bucket"], "item_id": row["item_id"]}
                    for row in time_tags
                ],
            },
        )
    raise ValueError(f"Unknown perturbation variant: {variant}")


def _sequence_artifact(
    examples: list[RankingExample],
    variants: list[str],
    *,
    popularity: Counter[str],
    seed: int,
) -> dict[str, Any]:
    users: list[dict[str, Any]] = []
    for example in examples:
        transformed: dict[str, list[str]] = {}
        for variant in variants:
            transformed[variant] = _transform_example_history(
                example,
                variant=variant,
                popularity=popularity,
                seed=seed,
            )[0]
        users.append(
            {
                "target_item": example.target_item,
                "transforms": transformed,
                "user_id": example.user_id,
            }
        )
    return {"variants": variants, "users": users}


def _prediction_overlap_rows(
    predictions: list[dict[str, Any]],
    *,
    variants: list[str],
    k: int,
) -> list[dict[str, Any]]:
    original = [row for row in predictions if row["metadata"]["perturbation"] == "original"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant == "original":
            continue
        variant_rows = [row for row in predictions if row["metadata"]["perturbation"] == variant]
        for method in sorted({str(row["method"]) for row in variant_rows}):
            stats = compare_prediction_sets(
                [row for row in original if str(row["method"]) == method],
                [row for row in variant_rows if str(row["method"]) == method],
                k=k,
            )
            rows.append({"method": method, "perturbation": variant, **stats})
    return rows
