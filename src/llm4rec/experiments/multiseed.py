"""Multi-seed smoke runner and aggregation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.experiments.runner import run_experiment
from llm4rec.io.artifacts import ensure_dir, write_csv_rows, write_json, write_yaml


def aggregate_seed_metrics(seed_metrics: list[dict[str, Any]], *, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate mean/std for metric values across seeds."""

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        values = [float(row.get(metric, 0.0) or 0.0) for row in seed_metrics]
        mean = sum(values) / float(len(values) or 1)
        variance = sum((value - mean) ** 2 for value in values) / float(max(len(values) - 1, 1))
        rows.append({"metric": metric, "mean": mean, "std": variance**0.5, "num_seeds": len(values)})
    return rows


def run_multiseed(config_path: str | Path) -> Path:
    """Run a small multi-seed config and export aggregate_metrics.csv."""

    config = load_yaml_config(config_path)
    seeds = [int(seed) for seed in config.get("seeds", config.get("experiment", {}).get("seeds", [2026]))]
    output_root = ensure_dir(resolve_path(config.get("experiment", {}).get("output_dir", "outputs/runs")))
    run_id = str(config.get("experiment", {}).get("run_id", "multiseed_smoke"))
    root = ensure_dir(output_root / run_id)
    base_config_path = config.get("base_config", "configs/experiments/phase2a_smoke.yaml")
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        base = load_yaml_config(base_config_path)
        run_config = deepcopy(base)
        run_config.setdefault("experiment", {})
        run_config["experiment"]["seed"] = seed
        run_config["experiment"]["run_id"] = f"{run_id}_seed_{seed}"
        run_config["experiment"]["output_dir"] = str(output_root)
        run_config["experiment"]["overwrite"] = True
        temp_config = root / f"seed_{seed}.yaml"
        write_yaml(temp_config, run_config)
        result = run_experiment(temp_config)
        overall = result.metrics.get("overall", {})
        seed_rows.append({"run_dir": str(result.run_dir), "seed": seed, **overall})
    metric_names = [str(name) for name in config.get("metrics", ["Recall@5", "NDCG@5", "MRR@5"])]
    aggregate = aggregate_seed_metrics(seed_rows, metric_names=metric_names)
    write_csv_rows(root / "seed_metrics.csv", seed_rows)
    write_csv_rows(root / "aggregate_metrics.csv", aggregate)
    write_json(root / "aggregate_metrics.json", {"aggregate": aggregate, "seeds": seed_rows})
    return root
