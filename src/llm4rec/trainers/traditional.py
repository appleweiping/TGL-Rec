"""Traditional baseline training for smoke-scale pre-experiment checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
from llm4rec.experiments.config import resolve_path, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl
from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.trainers.base import TrainingResult
from llm4rec.trainers.checkpointing import load_checkpoint, save_checkpoint
from llm4rec.utils.env import collect_environment


class TraditionalBaselineTrainer:
    """Train or load a small MF/BPR ranker and write shared prediction schema outputs."""

    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = config
        self.experiment = dict(config.get("experiment", {}))
        self.training = dict(config.get("training", {}))
        self.seed = int(self.training.get("seed", self.experiment.get("seed", 2026)))
        self.run_id = str(self.training.get("run_id", f"{self.experiment.get('run_id', 'smoke')}_train"))
        self.output_root = ensure_dir(resolve_path(self.experiment.get("output_dir", "outputs/runs")))
        self.run_dir = ensure_dir(self.output_root / self.run_id)
        self.artifacts_dir = ensure_dir(self.run_dir / "artifacts")
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        self.logger = RunLogger(self.run_dir / "logs.txt")

    def train(self) -> TrainingResult:
        set_global_seed(self.seed)
        save_resolved_config(self.config, self.run_dir / "training_config.yaml")
        save_resolved_config(self.config, self.run_dir / "resolved_config.yaml")
        write_json(self.run_dir / "environment.json", collect_environment())
        processed_dir = self._processed_dataset()
        checkpoint_path = self.checkpoint_dir / "mf_checkpoint.json"
        eval_only = bool(self.training.get("eval_only", False))
        resume = bool(self.training.get("resume", False))
        if eval_only:
            checkpoint = load_checkpoint(self.training.get("checkpoint_path", checkpoint_path))
        elif resume and checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path)
        else:
            checkpoint = self._train_mf(processed_dir, checkpoint_path)
        predictions = self._predict(processed_dir, checkpoint)
        write_jsonl(self.run_dir / "predictions.jsonl", predictions)
        metrics = {
            "checkpoint_path": str(checkpoint_path),
            "eval_only": eval_only,
            "method": checkpoint.get("method", "mf"),
            "num_predictions": len(predictions),
            "reportable": False,
            "seed": self.seed,
        }
        write_json(self.run_dir / "training_metrics.json", metrics)
        return TrainingResult(run_dir=self.run_dir, checkpoint_path=checkpoint_path, metrics=metrics)

    def _processed_dataset(self) -> Path:
        dataset = dict(self.config.get("dataset", {}))
        dataset["preprocess"] = True
        dataset["output_dir"] = str(self.artifacts_dir / "processed_dataset")
        result = preprocess_from_config({"dataset": dataset})
        return result.output_dir

    def _train_mf(self, processed_dir: Path, checkpoint_path: Path) -> dict[str, Any]:
        baseline = dict(self.training.get("baseline", {}))
        params = dict(baseline.get("params", {}))
        ranker = MatrixFactorizationRanker(
            factors=int(params.get("factors", 4)),
            epochs=int(params.get("epochs", 5)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            regularization=float(params.get("regularization", 0.002)),
            seed=self.seed,
        )
        train_rows = read_jsonl(processed_dir / "train.jsonl")
        item_rows = read_jsonl(processed_dir / "items.jsonl")
        ranker.fit(train_rows, item_rows)
        payload = {
            "item_factors": ranker.item_factors,
            "item_ids": ranker.item_ids,
            "item_popularity": dict(ranker.item_popularity),
            "method": "mf",
            "params": {
                "epochs": ranker.epochs,
                "factors": ranker.factors,
                "learning_rate": ranker.learning_rate,
                "regularization": ranker.regularization,
            },
            "seed": self.seed,
            "user_factors": ranker.user_factors,
            "user_positives": {user: sorted(items) for user, items in ranker.user_positives.items()},
        }
        return save_checkpoint(checkpoint_path, payload, overwrite=bool(self.training.get("overwrite", True))) and payload

    def _predict(self, processed_dir: Path, checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
        ranker = MatrixFactorizationRanker(
            factors=int(checkpoint.get("params", {}).get("factors", 4)),
            seed=int(checkpoint.get("seed", self.seed)),
        )
        ranker.item_ids = [str(item) for item in checkpoint.get("item_ids", [])]
        ranker.item_factors = {str(k): [float(x) for x in v] for k, v in checkpoint.get("item_factors", {}).items()}
        ranker.user_factors = {str(k): [float(x) for x in v] for k, v in checkpoint.get("user_factors", {}).items()}
        ranker.item_popularity.update({str(k): int(v) for k, v in checkpoint.get("item_popularity", {}).items()})
        interactions = read_jsonl(processed_dir / "interactions.jsonl")
        candidates = read_jsonl(processed_dir / "candidates.jsonl")
        predictions: list[dict[str, Any]] = []
        for row in candidates:
            if str(row.get("split")) != str(self.training.get("eval_split", "test")):
                continue
            example = _candidate_to_example(row, interactions)
            result = ranker.rank(example)
            predictions.append(
                {
                    "candidate_items": example.candidate_items,
                    "domain": example.domain,
                    "metadata": {"phase": "phase4_train_smoke", "reportable": False, **result.metadata},
                    "method": "mf",
                    "predicted_items": result.items,
                    "raw_output": result.raw_output,
                    "scores": result.scores,
                    "target_item": example.target_item,
                    "user_id": example.user_id,
                }
            )
        return predictions


def _candidate_to_example(candidate_row: dict[str, Any], interactions: list[dict[str, Any]]) -> RankingExample:
    user_id = str(candidate_row["user_id"])
    target_item = str(candidate_row["target_item"])
    history = [
        str(row["item_id"])
        for row in sorted(
            interactions,
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            ),
        )
        if str(row["user_id"]) == user_id and str(row.get("split")) != "test" and str(row["item_id"]) != target_item
    ]
    return RankingExample(
        user_id=user_id,
        history=history,
        target_item=target_item,
        candidate_items=[str(item) for item in candidate_row["candidate_items"]],
        domain=None if candidate_row.get("domain") is None else str(candidate_row["domain"]),
    )
