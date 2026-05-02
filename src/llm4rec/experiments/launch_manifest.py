"""Paper-scale launch manifest generation without execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.experiments.manifest import manifest_from_config
from llm4rec.experiments.protocol_version import DEFAULT_PAPER_CONFIGS
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.utils.env import current_git_commit


def create_launch_manifest(
    output_path: str | Path = "outputs/launch/paper_v1/launch_manifest.json",
    *,
    config_paths: list[str | Path] | None = None,
    protocol_version: str = "protocol_v1",
    owner: str = "research",
    notes: str = "Phase 8 launch package; no jobs executed.",
) -> dict[str, Any]:
    """Create the paper launch manifest and save it."""

    output = resolve_path(output_path)
    ensure_dir(output.parent)
    config_paths = config_paths or DEFAULT_PAPER_CONFIGS
    experiments = [_load_experiment(path) for path in config_paths]
    methods = sorted({method for experiment in experiments for method in experiment["methods"]})
    datasets = sorted({str(experiment["dataset"]) for experiment in experiments})
    seeds = sorted({seed for experiment in experiments for seed in experiment["seeds"]})
    total_runs = sum(len(experiment["methods"]) * len(experiment["seeds"]) for experiment in experiments)
    manifest = {
        NO_EXECUTION_FLAG: True,
        "api_calls_planned": 0,
        "code_commit": current_git_commit(resolve_path(".")),
        "datasets": datasets,
        "estimated_disk_gb": round(max(1, total_runs) * 0.08 + 5.0, 3),
        "estimated_gpu_hours": round(_sum_resource(experiments, "estimated_gpu_hours"), 3),
        "estimated_runtime_hours": round(_sum_resource(experiments, "estimated_runtime_hours"), 3),
        "experiment_configs": [experiment["config_path"] for experiment in experiments],
        "experiments": experiments,
        "expected_output_dirs": [experiment["output_dir"] for experiment in experiments],
        "failure_policy": "continue_and_audit",
        "launch_status": "planned",
        "lora_training_jobs_planned": 0,
        "methods": methods,
        "notes": notes,
        "owner": owner,
        "protocol_version": protocol_version,
        "resume_policy": "resume_from_checkpoints_and_predictions",
        "seeds": seeds,
        "table_export_plan": str(resolve_path("outputs/launch/paper_v1/table_plan.json")),
        "total_planned_runs": total_runs,
    }
    write_json(output, manifest)
    return manifest


def _load_experiment(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    config = load_yaml_config(resolved)
    manifest = manifest_from_config(config).data
    resource = dict(config.get("resource_budget", {}))
    methods = [str(method.get("name", method)) if isinstance(method, dict) else str(method) for method in manifest.get("methods", [])]
    return {
        "api_calls_allowed": bool(config.get("api_calls_allowed", config.get("llm", {}).get("allow_api_calls", False))),
        "candidate_artifact": config.get("candidate_artifact") or config.get("artifacts", {}).get("candidate_artifact"),
        "candidate_strategy": manifest.get("candidate_strategy"),
        "config_path": str(resolved),
        "dataset": manifest.get("dataset"),
        "estimated_disk_gb": float(resource.get("estimated_disk_gb", 1.0)),
        "estimated_gpu_hours": float(resource.get("estimated_gpu_hours", 0.0)),
        "estimated_runtime_hours": float(resource.get("estimated_runtime_hours", 1.0)),
        "failure_policy": config.get("failure_policy", "continue_and_audit"),
        "lora_training_enabled": bool(config.get("lora_training_enabled", config.get("training", {}).get("enable_lora_training", False))),
        "methods": methods,
        "metrics": [str(metric) for metric in manifest.get("metrics", [])],
        "output_dir": manifest.get("output_dir"),
        "protocol_version": config.get("protocol_version", manifest.get("protocol_version")),
        "reportable": bool(manifest.get("reportable")),
        "resume_policy": config.get("resume_policy", "resume"),
        "run_mode": manifest.get("run_mode"),
        "seeds": [int(seed) for seed in manifest.get("seeds", [])],
        "split_artifact": config.get("split_artifact") or config.get("artifacts", {}).get("split_artifact"),
        "split_strategy": manifest.get("split_strategy"),
    }


def _sum_resource(experiments: list[dict[str, Any]], key: str) -> float:
    return sum(float(experiment.get(key, 0.0)) for experiment in experiments)
