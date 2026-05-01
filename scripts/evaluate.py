"""Thin evaluation wrapper for llm4rec experiment configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.experiments.config import resolve_experiment_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = resolve_experiment_config(args.config)
    experiment = config.get("experiment", {})
    run_id = str(experiment.get("run_id", "phase1_smoke"))
    run_dir = resolve_path(experiment.get("output_dir", "outputs/runs")) / run_id
    evaluation = config.get("evaluation", {})
    dataset = config.get("dataset", {})
    predictions_path = resolve_path(evaluation.get("predictions_path", run_dir / "predictions.jsonl"))
    item_catalog_path = resolve_path(
        evaluation.get("item_catalog_path", run_dir / "artifacts" / "processed_dataset" / "items.jsonl")
    )
    metrics = evaluate_predictions(
        predictions_path=predictions_path,
        item_catalog_path=item_catalog_path,
        output_dir=run_dir,
        ks=tuple(int(k) for k in evaluation.get("ks", [1, 3, 5])),
        candidate_protocol=str(
            evaluation.get("candidate_protocol", dataset.get("candidate_protocol", "full_catalog"))
        ),
    )
    print(f"evaluated predictions: {predictions_path}")
    print(f"metrics written: {run_dir / 'metrics.json'}")
    print(f"num_predictions={metrics['num_predictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
