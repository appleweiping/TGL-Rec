from __future__ import annotations

import os
from pathlib import Path

from llm4rec.diagnostics.perturbation_runner import run_perturbation_experiment
from llm4rec.diagnostics.time_window_runner import run_time_window_experiment


def test_time_window_runner_writes_graph_summaries() -> None:
    root = Path(__file__).resolve().parents[2]
    run_id = f"window_unit_{os.getpid()}"
    config_dir = root / "outputs" / "test_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = f"""
experiment:
  run_id: {run_id}
  output_dir: outputs/test_runs
  seed: 2026
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
evaluation:
  ks: [1, 3, 5]
  candidate_protocol: full_catalog
  top_k: 5
"""
    perturb_config = config_dir / f"{run_id}_perturb.yaml"
    perturb_config.write_text(
        base
        + f"""
methods: [{root / "configs" / "baselines" / "popularity.yaml"}]
diagnostic:
  variants: [original]
""",
        encoding="utf-8",
    )
    window_config = config_dir / f"{run_id}_window.yaml"
    window_config.write_text(
        base
        + """
diagnostic:
  windows: [1h, 1d, 7d, 30d, full]
  half_life_seconds: 604800
  max_history_items: 5
""",
        encoding="utf-8",
    )
    run_perturbation_experiment(perturb_config)
    run_dir = run_time_window_experiment(window_config)
    assert (run_dir / "time_window_graph_summary.csv").is_file()
    assert (run_dir / "diagnostics" / "transition_edges.jsonl").is_file()
    assert (run_dir / "diagnostics" / "time_window_edges_1d.jsonl").is_file()
