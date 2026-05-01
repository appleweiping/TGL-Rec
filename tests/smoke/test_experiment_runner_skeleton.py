from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_experiment_runner_skeleton_writes_run_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    dataset_config = {
        "dataset": {
            "name": "tiny_runner",
            "adapter": "tiny_jsonl",
            "paths": {
                "interactions": str(root / "data" / "tiny" / "interactions.jsonl"),
                "items": str(root / "data" / "tiny" / "items.jsonl"),
            },
            "output_dir": str(tmp_path / "unused_processed"),
            "split_strategy": "leave_one_out",
            "candidate_protocol": "full_catalog",
            "seed": 2026,
        }
    }
    eval_config = {
        "evaluation": {
            "ks": [1, 3],
            "candidate_protocol": "full_catalog",
        }
    }
    dataset_config_path = tmp_path / "dataset.yaml"
    eval_config_path = tmp_path / "eval.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
    eval_config_path.write_text(yaml.safe_dump(eval_config), encoding="utf-8")
    experiment_config = {
        "experiment": {
            "run_id": "smoke_test",
            "output_dir": str(tmp_path / "runs"),
            "overwrite": True,
            "run_mode": "smoke",
            "seed": 2026,
        },
        "dataset": {"config_path": str(dataset_config_path), "preprocess": True},
        "method": {"name": "skeleton", "reportable": False, "eval_split": "test"},
        "evaluation": {"config_path": str(eval_config_path)},
    }
    experiment_config_path = tmp_path / "smoke.yaml"
    experiment_config_path.write_text(yaml.safe_dump(experiment_config), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_experiment.py"),
            "--config",
            str(experiment_config_path),
        ],
        cwd=root,
        check=True,
    )
    run_dir = tmp_path / "runs" / "smoke_test"
    expected = [
        "resolved_config.yaml",
        "environment.json",
        "logs.txt",
        "predictions.jsonl",
        "metrics.json",
        "metrics.csv",
    ]
    for name in expected:
        assert (run_dir / name).is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_predictions"] == 4
    assert metrics["overall"]["Recall@1"] == 1.0
