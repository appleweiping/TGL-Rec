"""Phase 5 smoke runners for TimeGraphEvidenceRec."""

from __future__ import annotations

import json
import shutil
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
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.methods.ablation import REQUIRED_ABLATIONS, build_ablation_configs, validate_ablation_names
from llm4rec.methods.config import resolve_method_from_experiment
from llm4rec.methods.method_card import render_method_card, write_method_card
from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.time_graph_evidence_ranker import (
    TimeGraphEvidenceRanker,
    prediction_row_from_result,
)
from llm4rec.utils.env import collect_environment


@dataclass(frozen=True)
class MethodSmokeResult:
    """Result returned by Phase 5 smoke runners."""

    run_dir: Path
    metrics: dict[str, Any]


def run_time_graph_evidence_smoke(config_path: str | Path) -> MethodSmokeResult:
    """Run method or ablation smoke config without API calls or real training."""

    config = resolve_experiment_config(config_path)
    experiment = dict(config.get("experiment", {}))
    if bool(experiment.get("ablation", False)) or "ablation" in str(experiment.get("run_id", "")):
        return _run_ablation_smoke(config)
    return _run_single_method_smoke(config)


def _run_single_method_smoke(config: dict[str, Any]) -> MethodSmokeResult:
    run_dir, artifacts_dir, logger = _prepare_run(config)
    processed_dir = _prepare_dataset(config, artifacts_dir, logger)
    method_config = resolve_method_from_experiment(config)
    method_config = _merge_method_overrides(method_config, config)
    method_config.setdefault("method", {})["reportable"] = False
    candidate_protocol = str(config.get("evaluation", {}).get("candidate_protocol", config["dataset"].get("candidate_protocol", "full_catalog")))
    predictions, evidence_rows = _run_predictions(
        processed_dir=processed_dir,
        method_config=method_config,
        candidate_protocol=candidate_protocol,
        phase="phase5_method_smoke",
        artifacts_dir=artifacts_dir / "method",
    )
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    write_jsonl(run_dir / "evidence_used.jsonl", evidence_rows)
    metrics = _evaluate(config, processed_dir=processed_dir, predictions_path=predictions_path, run_dir=run_dir)
    write_method_card(method_config, run_dir / "method_card.md")
    logger.info("phase5 method smoke completed")
    return MethodSmokeResult(run_dir=run_dir, metrics=metrics)


def _run_ablation_smoke(config: dict[str, Any]) -> MethodSmokeResult:
    run_dir, artifacts_dir, logger = _prepare_run(config)
    processed_dir = _prepare_dataset(config, artifacts_dir, logger)
    base_method_config = resolve_method_from_experiment(config)
    base_method_config = _merge_method_overrides(base_method_config, config)
    base_method_config.setdefault("method", {})["reportable"] = False
    ablation_section = dict(config.get("ablation", {}))
    names = [str(name) for name in ablation_section.get("names", REQUIRED_ABLATIONS)]
    validate_ablation_names(names)
    method_configs = build_ablation_configs(base_method_config, names=names)
    predictions_root = ensure_dir(run_dir / "predictions")
    metrics_root = ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "artifacts")
    candidate_protocol = str(config.get("evaluation", {}).get("candidate_protocol", config["dataset"].get("candidate_protocol", "full_catalog")))
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    for method_config in method_configs:
        name = str(method_config["method"]["ablation_name"])
        logger.info(f"running ablation={name}")
        predictions, evidence_rows = _run_predictions(
            processed_dir=processed_dir,
            method_config=method_config,
            candidate_protocol=candidate_protocol,
            phase="phase5_ablation_smoke",
            artifacts_dir=artifacts_dir / "ablations" / name,
        )
        predictions_path = predictions_root / f"{name}.jsonl"
        write_jsonl(predictions_path, predictions)
        write_jsonl(artifacts_dir / "evidence" / f"{name}.jsonl", evidence_rows)
        metrics_dir = ensure_dir(metrics_root / name)
        metrics = _evaluate(config, processed_dir=processed_dir, predictions_path=predictions_path, run_dir=metrics_dir)
        last_metrics = metrics
        overall = dict(metrics.get("overall", {}))
        summary_rows.append(
            {
                "ablation": name,
                "Recall@1": overall.get("Recall@1", 0.0),
                "NDCG@3": overall.get("NDCG@3", 0.0),
                "MRR@5": overall.get("MRR@5", 0.0),
                "evidence_rows": len(evidence_rows),
                "prediction_rows": len(predictions),
                "reportable": False,
            }
        )
        manifest_rows.append(
            {
                "ablation": name,
                "method_config": method_config,
                "metrics_path": str(metrics_dir / "metrics.json"),
                "predictions_path": str(predictions_path),
                "reportable": False,
            }
        )
    write_csv_rows(
        run_dir / "ablation_results.csv",
        summary_rows,
        fieldnames=["ablation", "Recall@1", "NDCG@3", "MRR@5", "evidence_rows", "prediction_rows", "reportable"],
    )
    write_json(
        run_dir / "ablation_manifest.json",
        {
            "ablations": manifest_rows,
            "reportable": False,
            "warning": "Phase 5 ablation smoke only; not paper-scale results.",
        },
    )
    logger.info("phase5 ablation smoke completed")
    return MethodSmokeResult(run_dir=run_dir, metrics=last_metrics)


def _prepare_run(config: dict[str, Any]) -> tuple[Path, Path, RunLogger]:
    experiment = dict(config.get("experiment", {}))
    seed = int(experiment.get("seed", 2026))
    set_global_seed(seed)
    run_id = str(experiment.get("run_id", "phase5_method_smoke"))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = output_root / run_id
    if run_dir.exists() and bool(experiment.get("overwrite", False)):
        _remove_run_dir(run_dir, output_root)
    ensure_dir(run_dir)
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    logger = RunLogger(run_dir / "logs.txt")
    logger.info(f"starting Phase 5 run_id={run_id}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    return run_dir, artifacts_dir, logger


def _prepare_dataset(config: dict[str, Any], artifacts_dir: Path, logger: RunLogger) -> Path:
    dataset_config = dict(config["dataset"])
    dataset_config["output_dir"] = str(artifacts_dir / "processed_dataset")
    logger.info("preprocessing smoke dataset")
    return preprocess_from_config({"dataset": dataset_config}).output_dir


def _run_predictions(
    *,
    processed_dir: Path,
    method_config: dict[str, Any],
    candidate_protocol: str,
    phase: str,
    artifacts_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    all_interactions = read_jsonl(processed_dir / "interactions.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    candidate_rows = read_jsonl(processed_dir / "candidates.jsonl")
    ranker = TimeGraphEvidenceRanker(method_config, candidate_protocol=candidate_protocol)
    ranker.fit(train_rows, item_rows)
    ranker.save_artifact(artifacts_dir)
    eval_split = str(method_config.get("method", {}).get("eval_split", "test"))
    method_name = _method_name(method_config)
    predictions: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for candidate_row in candidate_rows:
        if str(candidate_row["split"]) != eval_split:
            continue
        example = _candidate_to_example(candidate_row, all_interactions)
        result = ranker.rank(example)
        predictions.append(
            prediction_row_from_result(
                example=example,
                result=result,
                method_name=method_name,
                phase=phase,
            )
        )
        for evidence in result.metadata.get("evidence_used", []):
            evidence_rows.append(
                {
                    "ablation": method_config.get("method", {}).get("ablation_name"),
                    "candidate_items": example.candidate_items,
                    "evidence": evidence,
                    "target_item": example.target_item,
                    "user_id": example.user_id,
                }
            )
    return (
        sorted(predictions, key=lambda value: (str(value["method"]), str(value["user_id"]), str(value["target_item"]))),
        evidence_rows,
    )


def _evaluate(
    config: dict[str, Any],
    *,
    processed_dir: Path,
    predictions_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    return evaluate_predictions(
        predictions_path=predictions_path,
        item_catalog_path=processed_dir / "items.jsonl",
        output_dir=run_dir,
        ks=tuple(int(k) for k in evaluation.get("ks", [1, 3, 5])),
        candidate_protocol=str(evaluation.get("candidate_protocol", config["dataset"].get("candidate_protocol", "full_catalog"))),
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
        metadata={"prediction_timestamp": target_ts, "split": split},
    )


def _method_name(method_config: dict[str, Any]) -> str:
    method = dict(method_config.get("method", {}))
    base = str(method.get("name", "time_graph_evidence_rec"))
    ablation = method.get("ablation_name")
    return base if not ablation else f"{base}:{ablation}"


def _merge_method_overrides(method_config: dict[str, Any], experiment_config: dict[str, Any]) -> dict[str, Any]:
    import copy

    output = copy.deepcopy(method_config)
    for section in ("encoder", "ablation", "scoring", "retrieval", "translator"):
        if section in experiment_config:
            output.setdefault(section, {})
            if isinstance(output[section], dict) and isinstance(experiment_config[section], dict):
                output[section].update(experiment_config[section])
            else:
                output[section] = experiment_config[section]
    return output


def _remove_run_dir(run_dir: Path, output_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run == resolved_root or resolved_root not in resolved_run.parents:
        raise ValueError(f"Refusing to remove run dir outside output root: {run_dir}")
    try:
        shutil.rmtree(resolved_run)
    except FileNotFoundError:
        return


def export_method_card(config_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Export the Phase 5 method card from a method config."""

    config = load_yaml_config(config_path)
    output = Path(output_path) if output_path is not None else resolve_path("docs/method_card_time_graph_evidence.md")
    return write_method_card(config, output)


def method_card_text(config_path: str | Path) -> str:
    """Return rendered method-card text for tests."""

    return render_method_card(load_yaml_config(config_path))
