from __future__ import annotations

import json
import os
from pathlib import Path

from llm4rec.diagnostics.perturbation_runner import run_perturbation_experiment


def test_perturbation_runner_uses_shared_candidates() -> None:
    root = Path(__file__).resolve().parents[2]
    run_id = f"perturb_unit_{os.getpid()}"
    config_dir = root / "outputs" / "test_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / f"{run_id}.yaml"
    config.write_text(
        f"""
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
methods: [{root / "configs" / "baselines" / "popularity.yaml"}, {root / "configs" / "baselines" / "bm25.yaml"}]
evaluation:
  ks: [1, 3, 5]
  candidate_protocol: full_catalog
  top_k: 5
diagnostic:
  variants: [original, reversed, recent_5, timestamp_bucketed_prompt_ready]
""",
        encoding="utf-8",
    )
    run_dir = run_perturbation_experiment(config)
    assert (run_dir / "perturbation_results.csv").is_file()
    assert (run_dir / "perturbation_deltas.csv").is_file()
    rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert rows
    first = json.loads(rows[0])
    assert first["candidate_items"]
    assert first["metadata"]["perturbation"] in {
        "original",
        "reversed",
        "recent_5",
        "timestamp_bucketed_prompt_ready",
    }
