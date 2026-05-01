from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_phase2a_baselines_share_prediction_schema_and_evaluator() -> None:
    root = Path(__file__).resolve().parents[2]
    run_id = f"phase2a_test_{os.getpid()}"
    config_dir = root / "outputs" / "test_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}.yaml"
    config_path.write_text(
        f"""
experiment:
  run_id: {run_id}
  output_dir: outputs/test_runs
  overwrite: false
  run_mode: smoke
  seed: 2026
dataset:
  config_path: configs/datasets/tiny.yaml
  preprocess: true
methods: [configs/baselines/random.yaml, configs/baselines/popularity.yaml, configs/baselines/bm25.yaml, configs/baselines/mf.yaml]
evaluation:
  config_path: configs/evaluation/default.yaml
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_all.py"),
            "--config",
            str(config_path),
        ],
        cwd=root,
        check=True,
    )
    run_dir = root / "outputs" / "test_runs" / run_id
    predictions = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(predictions) == 16
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["by_method"]) == {"bm25", "mf", "popularity", "random"}
    assert (run_dir / "metrics.csv").is_file()
