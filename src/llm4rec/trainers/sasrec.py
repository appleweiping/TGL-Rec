"""Smoke-scale SASRec training and evaluation."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
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
from llm4rec.models.sasrec import SASRecModel, TORCH_AVAILABLE
from llm4rec.utils.env import collect_environment


@dataclass(frozen=True)
class SASRecRunResult:
    """Result of a SASRec smoke command."""

    run_dir: Path
    checkpoint_path: Path | None
    metrics: dict[str, Any]
    status: str


@dataclass(frozen=True)
class SASRecTrainingExample:
    """One next-item training example."""

    input_indices: list[int]
    positive_index: int
    negative_indices: list[int]


class SASRecSequenceDataset:
    """Fixed-length train-only next-item examples."""

    def __init__(
        self,
        *,
        train_interactions: list[dict[str, Any]],
        item_to_idx: dict[str, int],
        max_seq_len: int,
        num_negatives: int,
        seed: int,
    ) -> None:
        self.item_to_idx = dict(item_to_idx)
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        self.max_seq_len = int(max_seq_len)
        self.num_negatives = int(num_negatives)
        self.seed = int(seed)
        self.examples = self._build_examples(train_interactions)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SASRecTrainingExample:
        return self.examples[index]

    def _build_examples(self, interactions: list[dict[str, Any]]) -> list[SASRecTrainingExample]:
        sequences = build_user_sequences(interactions)
        all_items = sorted(self.item_to_idx)
        examples: list[SASRecTrainingExample] = []
        for user_id, items in sorted(sequences.items()):
            del user_id
            observed = {str(item) for item in items}
            for position in range(1, len(items)):
                history = [self.item_to_idx[item] for item in items[:position]][-self.max_seq_len :]
                target = self.item_to_idx[items[position]]
                negatives = sample_negative_items(
                    all_items=all_items,
                    positives=observed,
                    num_negatives=self.num_negatives,
                    seed=self.seed + len(examples),
                )
                examples.append(
                    SASRecTrainingExample(
                        input_indices=left_pad(history, self.max_seq_len),
                        positive_index=target,
                        negative_indices=[self.item_to_idx[item] for item in negatives],
                    )
                )
        return examples


def build_item_mappings(item_records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    """Map item ids to SASRec embedding ids, reserving 0 for padding."""

    item_to_idx = {str(row["item_id"]): index for index, row in enumerate(sorted(item_records, key=lambda row: str(row["item_id"])), start=1)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    return item_to_idx, idx_to_item


def build_user_sequences(interactions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build timestamp-ordered user sequences."""

    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in interactions:
        by_user.setdefault(str(row["user_id"]), []).append(row)
    sequences: dict[str, list[str]] = {}
    for user_id, rows in by_user.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            ),
        )
        sequences[user_id] = [str(row["item_id"]) for row in ordered]
    return sequences


def left_pad(values: list[int], max_seq_len: int) -> list[int]:
    """Left-pad a sequence with 0."""

    clipped = list(values)[-int(max_seq_len) :]
    return [0] * (int(max_seq_len) - len(clipped)) + clipped


def sample_negative_items(
    *,
    all_items: list[str],
    positives: set[str],
    num_negatives: int,
    seed: int,
) -> list[str]:
    """Deterministically sample negatives not observed by the user."""

    pool = [item for item in sorted(all_items) if item not in positives]
    if not pool:
        raise ValueError("negative sampling pool is empty")
    rng = random.Random(int(seed))
    return [pool[rng.randrange(len(pool))] for _ in range(int(num_negatives))]


def run_sasrec_smoke(config_path: str | Path) -> SASRecRunResult:
    """Run SASRec smoke training/evaluation or write a clear skip artifact."""

    config = _resolve_config(config_path)
    run_dir, artifacts_dir, logger = _prepare_run(config)
    if not TORCH_AVAILABLE:
        checkpoint_dir = ensure_dir(run_dir / "checkpoints")
        metrics = {
            "pytorch_available": False,
            "status": "skipped_pytorch_unavailable",
            "message": "Install the project with `.[models]` to enable SASRec smoke training.",
        }
        write_jsonl(run_dir / "predictions.jsonl", [])
        write_json(run_dir / "metrics.json", metrics)
        write_csv_rows(run_dir / "metrics.csv", [{"metric": key, "value": value} for key, value in metrics.items()])
        write_json(run_dir / "training_metrics.json", metrics)
        logger.info(metrics["message"])
        return SASRecRunResult(run_dir=run_dir, checkpoint_path=None, metrics=metrics, status="skipped_pytorch_unavailable")

    import torch
    from torch.utils.data import DataLoader

    processed_dir = _prepare_dataset(config, artifacts_dir, logger)
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    item_to_idx, idx_to_item = build_item_mappings(item_rows)
    training = dict(config.get("training", {}))
    seed = int(training.get("seed", config.get("experiment", {}).get("seed", 2026)))
    set_global_seed(seed)
    device = torch.device(_device_name(training.get("device", "cpu")))
    dataset = SASRecSequenceDataset(
        train_interactions=train_rows,
        item_to_idx=item_to_idx,
        max_seq_len=int(training.get("max_seq_len", 20)),
        num_negatives=int(training.get("num_negatives", 1)),
        seed=seed,
    )
    model = SASRecModel(
        num_items=len(item_to_idx),
        hidden_dim=int(training.get("hidden_dim", 32)),
        num_layers=int(training.get("num_layers", 1)),
        num_heads=int(training.get("num_heads", 1)),
        dropout=float(training.get("dropout", 0.1)),
        max_seq_len=int(training.get("max_seq_len", 20)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate_sasrec,
    )
    losses: list[float] = []
    model.train()
    for _epoch in range(int(training.get("epochs", 1))):
        for batch in loader:
            seq = batch["input"].to(device)
            pos = batch["positive"].to(device).unsqueeze(1)
            neg = batch["negative"].to(device)
            pos_scores = model.score_items(seq, pos)
            neg_scores = model.score_items(seq, neg)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    checkpoint_path = checkpoint_dir / "sasrec.pt"
    torch.save(
        {
            "config": config,
            "idx_to_item": idx_to_item,
            "item_to_idx": item_to_idx,
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )
    predictions = _sasrec_predictions(
        model=model,
        processed_dir=processed_dir,
        item_to_idx=item_to_idx,
        device=device,
        max_seq_len=int(training.get("max_seq_len", 20)),
    )
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
    logger.info("SASRec smoke training completed")
    return SASRecRunResult(run_dir=run_dir, checkpoint_path=checkpoint_path, metrics=metrics, status="trained")


def _sasrec_predictions(
    *,
    model: Any,
    processed_dir: Path,
    item_to_idx: dict[str, int],
    device: Any,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    all_interactions = read_jsonl(processed_dir / "interactions.jsonl")
    candidate_rows = read_jsonl(processed_dir / "candidates.jsonl")
    predictions: list[dict[str, Any]] = []
    for row in candidate_rows:
        if str(row["split"]) != "test":
            continue
        history = _history_before_target(row, all_interactions)
        history_indices = [item_to_idx[item] for item in history if item in item_to_idx]
        seq = torch.tensor([left_pad(history_indices, max_seq_len)], dtype=torch.long, device=device)
        candidate_items = [str(item) for item in row["candidate_items"]]
        candidate_indices = torch.tensor(
            [[item_to_idx.get(item, 0) for item in candidate_items]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            scores = model.score_items(seq, candidate_indices).squeeze(0).detach().cpu().tolist()
        scored = sorted(zip(candidate_items, scores), key=lambda item: (-float(item[1]), item[0]))
        predictions.append(
            {
                "candidate_items": candidate_items,
                "domain": row.get("domain"),
                "metadata": {"phase": "phase6_sasrec_smoke", "reportable": False},
                "method": "sasrec",
                "predicted_items": [item for item, _score in scored],
                "raw_output": None,
                "scores": [float(score) for _item, score in scored],
                "target_item": str(row["target_item"]),
                "user_id": str(row["user_id"]),
            }
        )
    return sorted(predictions, key=lambda value: (value["domain"] or "", value["user_id"], value["target_item"]))


def _history_before_target(candidate_row: dict[str, Any], interactions: list[dict[str, Any]]) -> list[str]:
    user_id = str(candidate_row["user_id"])
    target_item = str(candidate_row["target_item"])
    split = str(candidate_row["split"])
    targets = [
        row
        for row in interactions
        if str(row["user_id"]) == user_id
        and str(row["item_id"]) == target_item
        and str(row.get("split")) == split
    ]
    target_ts = float(targets[-1]["timestamp"]) if targets and targets[-1].get("timestamp") is not None else None
    history = []
    for row in interactions:
        if str(row["user_id"]) != user_id or str(row.get("split")) == "test":
            continue
        timestamp = None if row.get("timestamp") is None else float(row["timestamp"])
        if target_ts is not None and timestamp is not None and timestamp >= target_ts:
            continue
        history.append(row)
    return [
        str(row["item_id"])
        for row in sorted(
            history,
            key=lambda value: (
                float(value["timestamp"]) if value.get("timestamp") is not None else -1.0,
                str(value["item_id"]),
            ),
        )
    ]


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
    run_dir = output_root / str(experiment.get("run_id", "phase6_sasrec_smoke"))
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
    logger.info("preprocessing SASRec smoke dataset")
    return preprocess_from_config({"dataset": dataset_config}).output_dir


def _device_name(value: Any) -> str:
    if str(value) == "cuda":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def _collate_sasrec(batch: list[SASRecTrainingExample]) -> dict[str, Any]:
    import torch

    return {
        "input": torch.tensor([row.input_indices for row in batch], dtype=torch.long),
        "positive": torch.tensor([row.positive_index for row in batch], dtype=torch.long),
        "negative": torch.tensor([row.negative_indices for row in batch], dtype=torch.long),
    }


def _remove_run_dir(run_dir: Path, output_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run == resolved_root or resolved_root not in resolved_run.parents:
        raise ValueError(f"Refusing to remove run dir outside output root: {run_dir}")
    try:
        shutil.rmtree(resolved_run)
    except FileNotFoundError:
        return
