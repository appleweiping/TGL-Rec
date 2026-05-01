"""Config-driven experiment runner for smoke and Phase 2A baselines."""

from __future__ import annotations

import shutil
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.experiments.config import (
    load_yaml_config,
    resolve_experiment_config,
    resolve_path,
    save_resolved_config,
)
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl
from llm4rec.rankers.base import BaseRanker, RankingExample
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker
from llm4rec.rankers.sequential import GRU4RecInterface, MarkovTransitionRanker, SASRecInterface
from llm4rec.utils.env import collect_environment


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]


def run_experiment(config_path: str | Path) -> RunResult:
    """Run a configured smoke experiment."""

    config = resolve_experiment_config(config_path)
    if str(config.get("experiment", {}).get("stage", "")) == "phase2b":
        from llm4rec.diagnostics.diagnostic_export import run_phase2b_from_config

        run_dir = run_phase2b_from_config(config_path)
        metrics_path = run_dir / "diagnostic_summary.json"
        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return RunResult(run_dir=run_dir, metrics=metrics)
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", config.get("dataset", {}).get("seed", 0)))
    set_global_seed(seed)
    run_id = str(experiment.get("run_id", "phase1_smoke"))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = output_root / run_id
    cleanup_warning: str | None = None
    if run_dir.exists() and bool(experiment.get("overwrite", False)):
        try:
            _remove_run_dir(run_dir, output_root)
        except PermissionError as exc:
            cleanup_warning = f"could not fully remove old run dir before overwrite: {exc}"
    ensure_dir(run_dir)
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    logger = RunLogger(run_dir / "logs.txt")
    logger.info(f"starting run_id={run_id}")
    if cleanup_warning is not None:
        logger.info(cleanup_warning)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())

    dataset_config = dict(config["dataset"])
    if bool(dataset_config.get("preprocess", True)):
        dataset_config["output_dir"] = str(artifacts_dir / "processed_dataset")
        logger.info("running tiny preprocessing")
        preprocess_result = preprocess_from_config({"dataset": dataset_config})
        processed_dir = preprocess_result.output_dir
    else:
        processed_dir = resolve_path(dataset_config["output_dir"])
        logger.info(f"using existing processed dataset: {processed_dir}")

    predictions_path = run_dir / "predictions.jsonl"
    if config.get("methods"):
        method_configs = _resolve_method_configs(config.get("methods", []))
        logger.info(f"running {len(method_configs)} Phase 2A baseline rankers")
        predictions = _baseline_predictions(
            processed_dir=processed_dir,
            method_configs=method_configs,
            seed=seed,
            artifacts_dir=artifacts_dir,
        )
    else:
        logger.info("running deterministic non-reportable skeleton ranker")
        predictions = _skeleton_predictions(
            candidates_path=processed_dir / "candidates.jsonl",
            split=str(config.get("method", {}).get("eval_split", "test")),
            method_name=str(config.get("method", {}).get("name", "skeleton")),
        )
    write_jsonl(predictions_path, predictions)

    evaluation = config.get("evaluation", {})
    ks = tuple(int(k) for k in evaluation.get("ks", [1, 3, 5]))
    candidate_protocol = str(
        evaluation.get("candidate_protocol", dataset_config.get("candidate_protocol", "full_catalog"))
    )
    logger.info("running evaluator")
    metrics = evaluate_predictions(
        predictions_path=predictions_path,
        item_catalog_path=processed_dir / "items.jsonl",
        output_dir=run_dir,
        ks=ks,
        candidate_protocol=candidate_protocol,
    )
    logger.info("run completed")
    return RunResult(run_dir=run_dir, metrics=metrics)


def _skeleton_predictions(
    *,
    candidates_path: Path,
    split: str,
    method_name: str,
) -> list[dict[str, Any]]:
    candidate_rows = read_jsonl(candidates_path)
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if row["split"] != split:
            continue
        candidates = [str(item) for item in row["candidate_items"]]
        target = str(row["target_item"])
        rest = [item for item in candidates if item != target]
        predicted = [target, *rest]
        scores = [float(len(predicted) - index) for index, _ in enumerate(predicted)]
        rows.append(
            {
                "candidate_items": candidates,
                "domain": row.get("domain"),
                "metadata": {
                    "non_reportable": True,
                    "phase": "phase1_smoke",
                    "skeleton_ranker": "target_first_deterministic",
                },
                "method": method_name,
                "predicted_items": predicted,
                "raw_output": None,
                "scores": scores,
                "target_item": target,
                "user_id": str(row["user_id"]),
            }
        )
    return sorted(rows, key=lambda value: (value["domain"] or "", value["user_id"], value["target_item"]))


def _baseline_predictions(
    *,
    processed_dir: Path,
    method_configs: list[dict[str, Any]],
    seed: int,
    artifacts_dir: Path,
) -> list[dict[str, Any]]:
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    all_interactions = read_jsonl(processed_dir / "interactions.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    candidate_rows = read_jsonl(processed_dir / "candidates.jsonl")
    predictions: list[dict[str, Any]] = []
    for method_config in method_configs:
        ranker = _make_ranker(method_config, seed=seed)
        method_name = str(method_config.get("name", getattr(ranker, "name", "unknown")))
        eval_split = str(method_config.get("eval_split", "test"))
        ranker.fit(train_rows, item_rows)
        ranker.save_artifact(artifacts_dir / "methods" / method_name)
        for candidate_row in candidate_rows:
            if str(candidate_row["split"]) != eval_split:
                continue
            example = _candidate_to_example(candidate_row, all_interactions)
            result = ranker.rank(example)
            predictions.append(
                {
                    "candidate_items": example.candidate_items,
                    "domain": example.domain,
                    "metadata": {
                        **result.metadata,
                        "eval_split": eval_split,
                        "phase": "phase2a_smoke",
                        "reportable": bool(method_config.get("reportable", True)),
                    },
                    "method": method_name,
                    "predicted_items": result.items,
                    "raw_output": result.raw_output,
                    "scores": result.scores,
                    "target_item": example.target_item,
                    "user_id": example.user_id,
                }
            )
    return sorted(
        predictions,
        key=lambda value: (
            str(value["method"]),
            value["domain"] or "",
            value["user_id"],
            value["target_item"],
        ),
    )


def _candidate_to_example(
    candidate_row: dict[str, Any],
    all_interactions: list[dict[str, Any]],
) -> RankingExample:
    user_id = str(candidate_row["user_id"])
    target_item = str(candidate_row["target_item"])
    split = str(candidate_row["split"])
    target_rows = [
        row
        for row in all_interactions
        if str(row["user_id"]) == user_id
        and str(row["item_id"]) == target_item
        and str(row.get("split")) == split
    ]
    if not target_rows:
        raise ValueError(f"Could not find target interaction for {user_id}/{target_item}/{split}")
    target = sorted(
        target_rows,
        key=lambda row: float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
    )[-1]
    target_ts = None if target.get("timestamp") is None else float(target["timestamp"])
    history_rows = []
    for row in all_interactions:
        if str(row["user_id"]) != user_id:
            continue
        if row is target:
            continue
        timestamp = None if row.get("timestamp") is None else float(row["timestamp"])
        if target_ts is not None and timestamp is not None and timestamp >= target_ts:
            continue
        if str(row.get("split")) == "test":
            continue
        history_rows.append(row)
    history = [
        str(row["item_id"])
        for row in sorted(
            history_rows,
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            ),
        )
    ]
    return RankingExample(
        user_id=user_id,
        history=history,
        target_item=target_item,
        candidate_items=[str(item) for item in candidate_row["candidate_items"]],
        domain=None if candidate_row.get("domain") is None else str(candidate_row.get("domain")),
        metadata={"split": split},
    )


def _resolve_method_configs(methods: list[Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for method in methods:
        if isinstance(method, str):
            config = load_yaml_config(method)
            configs.append(dict(config.get("baseline", config)))
        elif isinstance(method, dict) and method.get("config_path"):
            config = load_yaml_config(method["config_path"])
            configs.append({**dict(config.get("baseline", config)), **{k: v for k, v in method.items() if k != "config_path"}})
        elif isinstance(method, dict):
            configs.append(dict(method))
        else:
            raise ValueError(f"Unsupported method config entry: {method!r}")
    return configs


def _make_ranker(config: dict[str, Any], *, seed: int) -> BaseRanker:
    method_type = str(config.get("type", config.get("name", ""))).lower()
    params = dict(config.get("params", {}))
    if method_type == "random":
        return RandomRanker(seed=int(params.get("seed", seed)))
    if method_type == "popularity":
        return PopularityRanker()
    if method_type == "bm25":
        return BM25Ranker(k1=float(params.get("k1", 1.5)), b=float(params.get("b", 0.75)))
    if method_type in {"mf", "bpr", "bpr_mf"}:
        return MatrixFactorizationRanker(
            factors=int(params.get("factors", 8)),
            epochs=int(params.get("epochs", 20)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            regularization=float(params.get("regularization", 0.002)),
            seed=int(params.get("seed", seed)),
        )
    if method_type in {"markov", "markov_transition", "sequential_markov"}:
        return MarkovTransitionRanker()
    if method_type == "sasrec":
        return SASRecInterface(reportable=bool(config.get("reportable", False)))
    if method_type == "gru4rec":
        return GRU4RecInterface(reportable=bool(config.get("reportable", False)))
    raise ValueError(f"Unknown ranker type: {method_type}")


def _remove_run_dir(run_dir: Path, output_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run == resolved_root or resolved_root not in resolved_run.parents:
        raise ValueError(f"Refusing to remove run dir outside output root: {run_dir}")
    shutil.rmtree(resolved_run)
