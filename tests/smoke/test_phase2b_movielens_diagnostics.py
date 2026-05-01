from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_phase2b_movielens_diagnostics_smoke_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    run_id = f"phase2b_fixture_{os.getpid()}"
    config_dir = root / "outputs" / "test_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = config_dir / f"{run_id}_dataset.yaml"
    perturb_config = config_dir / f"{run_id}_perturb.yaml"
    window_config = config_dir / f"{run_id}_windows.yaml"
    sim_config = config_dir / f"{run_id}_similarity.yaml"
    experiment_config = config_dir / f"{run_id}_experiment.yaml"
    dataset_config.write_text(
        f"""
dataset:
  name: fixture_ml
  adapter: movielens_style
  paths:
    raw_dir: data/fixtures/movielens_style
  output_dir: outputs/test_runs/{run_id}/processed
  split_strategy: leave_one_out
  candidate_protocol: full_catalog
  min_user_interactions: 3
  seed: 2026
""",
        encoding="utf-8",
    )
    perturb_config.write_text(
        f"""
experiment:
  run_id: {run_id}
  output_dir: outputs/test_runs
  seed: 2026
dataset:
  config_path: {dataset_config}
  preprocess: true
methods: [configs/baselines/popularity.yaml, configs/baselines/bm25.yaml]
evaluation:
  config_path: configs/evaluation/default.yaml
  top_k: 5
diagnostic:
  variants: [original, reversed, recent_5, popularity_sorted, timestamp_bucketed_prompt_ready]
""",
        encoding="utf-8",
    )
    window_config.write_text(
        f"""
experiment:
  run_id: {run_id}
  output_dir: outputs/test_runs
  seed: 2026
dataset:
  config_path: {dataset_config}
  preprocess: true
evaluation:
  config_path: configs/evaluation/default.yaml
  top_k: 5
diagnostic:
  windows: [1h, 1d, 7d, 30d, full]
  half_life_seconds: 604800
  max_history_items: 5
""",
        encoding="utf-8",
    )
    sim_config.write_text(
        """
diagnostic:
  similarity_threshold: 0.2
  top_k_pairs: 50
""",
        encoding="utf-8",
    )
    experiment_config.write_text(
        f"""
experiment:
  stage: phase2b
  run_id: {run_id}
  output_dir: outputs/test_runs
  seed: 2026
perturbation_config: {perturb_config}
time_window_config: {window_config}
similarity_config: {sim_config}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(root / "scripts" / "run_all.py"), "--config", str(experiment_config)],
        cwd=root,
        check=True,
    )
    run_dir = root / "outputs" / "test_runs" / run_id
    expected = [
        "diagnostic_summary.json",
        "perturbation_results.csv",
        "perturbation_deltas.csv",
        "prediction_overlap.csv",
        "time_window_graph_summary.csv",
        "similarity_vs_transition.json",
        "similarity_vs_transition.csv",
    ]
    for name in expected:
        assert (run_dir / name).is_file()
    summary = json.loads((run_dir / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert "strongest_perturbation_effect" in summary
    assert summary["similarity_vs_transition_counts"]["semantic_only"] >= 0
