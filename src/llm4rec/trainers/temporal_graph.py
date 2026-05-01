"""Smoke-scale training for the lightweight TemporalGraphEncoder."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
from llm4rec.encoders.temporal_graph_encoder import (
    TORCH_AVAILABLE,
    TemporalGraphEncoder,
    build_temporal_graph_mappings,
)
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.experiments.config import (
    deep_merge,
    load_yaml_config,
    resolve_experiment_config,
    resolve_path,
    save_resolved_config,
)
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.utils.env import collect_environment


@dataclass(frozen=True)
class TemporalGraphRunResult:
    """Result of temporal graph smoke training."""

    run_dir: Path
    checkpoint_path: Path | None
    metrics: dict[str, Any]
    status: str


def run_temporal_graph_smoke(config_path: str | Path) -> TemporalGraphRunResult:
    """Run temporal graph smoke training/evaluation or write a clear skip artifact."""

    config = _resolve_config(config_path)
    run_dir, artifacts_dir, logger = _prepare_run(config)
    if not TORCH_AVAILABLE:
        ensure_dir(run_dir / "checkpoints")
        metrics = {
            "pytorch_available": False,
            "status": "skipped_pytorch_unavailable",
            "message": "Install the project with `.[models]` to enable TemporalGraphEncoder training.",
        }
        write_jsonl(run_dir / "predictions.jsonl", [])
        write_json(run_dir / "metrics.json", metrics)
        write_csv_rows(run_dir / "metrics.csv", [{"metric": key, "value": value} for key, value in metrics.items()])
        write_json(run_dir / "training_metrics.json", metrics)
        logger.info(metrics["message"])
        return TemporalGraphRunResult(run_dir=run_dir, checkpoint_path=None, metrics=metrics, status="skipped_pytorch_unavailable")

    import torch

    processed_dir = _prepare_dataset(config, artifacts_dir, logger)
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    training = dict(config.get("training", {}))
    seed = int(training.get("seed", config.get("experiment", {}).get("seed", 2026)))
    set_global_seed(seed)
    user_to_idx, item_to_idx = build_temporal_graph_mappings(train_rows, item_rows)
    device = torch.device(_device_name(training.get("device", "cpu")))
    model = TemporalGraphEncoder(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        hidden_dim=int(training.get("hidden_dim", 16)),
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    events = sorted(train_rows, key=lambda row: (float(row.get("timestamp") or -1.0), str(row["user_id"]), str(row["item_id"])))
    all_items = sorted(item_to_idx)
    losses: list[float] = []
    for _epoch in range(int(training.get("epochs", 1))):
        rng = random.Random(seed + _epoch)
        for event in events:
            user_id = str(event["user_id"])
            item_id = str(event["item_id"])
            negatives = _sample_negatives(all_items, {item_id}, int(training.get("num_negatives", 1)), rng)
            user_idx = torch.tensor([user_to_idx[user_id]] * len(negatives), dtype=torch.long, device=device)
            pos_idx = torch.tensor([item_to_idx[item_id]] * len(negatives), dtype=torch.long, device=device)
            neg_idx = torch.tensor([item_to_idx[item] for item in negatives], dtype=torch.long, device=device)
            timestamp = torch.tensor([float(event.get("timestamp") or 0.0)] * len(negatives), dtype=torch.float32, device=device)
            pos_score = model.score_tensor(user_idx, pos_idx, timestamp)
            neg_score = model.score_tensor(user_idx, neg_idx, timestamp)
            loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update(event)
            losses.append(float(loss.detach().cpu().item()))
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    checkpoint_path = checkpoint_dir / "temporal_graph_encoder.pt"
    model.save(checkpoint_path)
    predictions = _temporal_graph_predictions(model=model, processed_dir=processed_dir)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    evaluation = dict(config.get("evaluation", {}))
    metrics = evaluate_predictions(
        predictions_path=predictions_path,
        item_catalog_path=processed_dir / "items.jsonl",
        output_dir=run_dir,
        ks=tuple(int(k) for k in evaluation.get("ks", [1, 3, 5])),
        candidate_protocol=str(evaluation.get("candidate_protocol", config["dataset"].get("candidate_protocol", "full_catalog"))),
    )
    training_metrics = {
        "checkpoint_path": str(checkpoint_path),
        "epochs": int(training.get("epochs", 1)),
        "final_loss": losses[-1] if losses else None,
        "loss_count": len(losses),
        "pytorch_available": True,
        "status": "trained",
    }
    write_json(run_dir / "training_metrics.json", training_metrics)
    logger.info("TemporalGraphEncoder smoke training completed")
    return TemporalGraphRunResult(run_dir=run_dir, checkpoint_path=checkpoint_path, metrics=metrics, status="trained")


def _temporal_graph_predictions(*, model: Any, processed_dir: Path) -> list[dict[str, Any]]:
    all_interactions = read_jsonl(processed_dir / "interactions.jsonl")
    candidate_rows = read_jsonl(processed_dir / "candidates.jsonl")
    predictions: list[dict[str, Any]] = []
    for row in candidate_rows:
        if str(row["split"]) != "test":
            continue
        timestamp = _prediction_timestamp(row, all_interactions)
        candidate_items = [str(item) for item in row["candidate_items"]]
        scores = [model.score(str(row["user_id"]), item, timestamp) for item in candidate_items]
        scored = sorted(zip(candidate_items, scores), key=lambda pair: (-float(pair[1]), pair[0]))
        predictions.append(
            {
                "candidate_items": candidate_items,
                "domain": row.get("domain"),
                "metadata": {"phase": "phase6_temporal_graph_smoke", "reportable": False},
                "method": "temporal_graph_encoder",
                "predicted_items": [item for item, _score in scored],
                "raw_output": None,
                "scores": [float(score) for _item, score in scored],
                "target_item": str(row["target_item"]),
                "user_id": str(row["user_id"]),
            }
        )
    return sorted(predictions, key=lambda value: (value["domain"] or "", value["user_id"], value["target_item"]))


def _prediction_timestamp(candidate_row: dict[str, Any], interactions: list[dict[str, Any]]) -> float | None:
    for row in interactions:
        if (
            str(row["user_id"]) == str(candidate_row["user_id"])
            and str(row["item_id"]) == str(candidate_row["target_item"])
            and str(row.get("split")) == str(candidate_row["split"])
        ):
            return None if row.get("timestamp") is None else float(row["timestamp"])
    return None


def _sample_negatives(all_items: list[str], positives: set[str], count: int, rng: random.Random) -> list[str]:
    pool = [item for item in all_items if item not in positives]
    if not pool:
        raise ValueError("negative sampling pool is empty")
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def _resolve_config(config_path: str | Path) -> dict[str, Any]:
    config = resolve_experiment_config(config_path)
    training = dict(config.get("training", {}))
    if training.get("config_path"):
        loaded = load_yaml_config(training["config_path"])
        config["training"] = deep_merge(
            dict(loaded.get("training", loaded)),
            {key: value for key, value in training.items() if key != "config_path"},
        )
    return config


def _prepare_run(config: dict[str, Any]) -> tuple[Path, Path, RunLogger]:
    experiment = dict(config.get("experiment", {}))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = output_root / str(experiment.get("run_id", "phase6_temporal_graph_smoke"))
    if run_dir.exists() and bool(experiment.get("overwrite", False)):
        _remove_run_dir(run_dir, output_root)
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    logger = RunLogger(run_dir / "logs.txt")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    return run_dir, artifacts_dir, logger


def _prepare_dataset(config: dict[str, Any], artifacts_dir: Path, logger: RunLogger) -> Path:
    dataset_config = dict(config["dataset"])
    dataset_config["output_dir"] = str(artifacts_dir / "processed_dataset")
    logger.info("preprocessing TemporalGraphEncoder smoke dataset")
    return preprocess_from_config({"dataset": dataset_config}).output_dir


def _device_name(value: Any) -> str:
    if str(value) == "cuda":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def _remove_run_dir(run_dir: Path, output_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run == resolved_root or resolved_root not in resolved_run.parents:
        raise ValueError(f"Refusing to remove run dir outside output root: {run_dir}")
    try:
        shutil.rmtree(resolved_run)
    except FileNotFoundError:
        return
