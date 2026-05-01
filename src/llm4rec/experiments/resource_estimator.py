"""Resource estimates for Phase 7 pilot configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.experiments.config import resolve_experiment_config, resolve_path, save_resolved_config
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.utils.env import collect_environment


def estimate_pilot_resources(config_path: str | Path) -> dict[str, Any]:
    """Estimate pilot runtime/resource envelope and save it before execution."""

    config = resolve_experiment_config(config_path)
    experiment = dict(config.get("experiment", {}))
    pilot = dict(config.get("pilot", {}))
    methods = list(config.get("methods", []))
    run_dir = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")) / str(experiment.get("run_id")))
    requested_users = int(pilot.get("max_users", 200))
    requested_items = int(pilot.get("max_items", 1000))
    requested_interactions = int(pilot.get("max_interactions", 10000))
    candidate_size = int(pilot.get("candidate_size", 100))
    estimated_cases = requested_users
    estimate = {
        "allow_api_calls": bool(config.get("llm", {}).get("allow_api_calls", False)),
        "candidate_size": candidate_size,
        "enable_lora_training": bool(config.get("training", {}).get("enable_lora_training", False)),
        "estimated_api_calls": 0,
        "estimated_candidate_scores": estimated_cases * candidate_size * max(1, len(methods)),
        "estimated_methods": len(methods),
        "estimated_prediction_cases": estimated_cases,
        "max_interactions": requested_interactions,
        "max_items": requested_items,
        "max_users": requested_users,
        "memory_estimate_mb": round(64 + requested_items * 0.02 + requested_users * 0.01, 3),
        "model_downloads": 0,
        "non_reportable_marker": "NON_REPORTABLE",
        "output_run_dir": str(run_dir),
        "pilot_reportable": bool(config.get("pilot_reportable", False)),
        "run_mode": str(experiment.get("run_mode", "")),
    }
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    write_json(run_dir / "resource_estimate.json", estimate)
    return estimate
