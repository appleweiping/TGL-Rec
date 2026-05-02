"""Conservative paper launch resource budget estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json

TRAINABLE_METHODS = {"mf", "bpr", "sasrec", "temporal_graph_encoder", "time_graph_evidence_dynamic"}


def estimate_paper_resources(
    manifest: dict[str, Any],
    output_path: str | Path = "outputs/launch/paper_v1/resource_budget.json",
) -> dict[str, Any]:
    """Estimate paper-scale resources from the launch manifest."""

    jobs = _planned_jobs(manifest)
    trainable = [job for job in jobs if job["method"] in TRAINABLE_METHODS]
    eval_only = [job for job in jobs if job["method"] not in TRAINABLE_METHODS]
    cpu_hours = len(eval_only) * 1.0 + len(trainable) * 1.5
    gpu_hours = sum(2.0 if job["method"] in {"sasrec", "temporal_graph_encoder", "time_graph_evidence_dynamic"} else 0.5 for job in trainable)
    prediction_gb = max(0.1, len(jobs) * 0.02)
    checkpoint_gb = max(0.1, len(trainable) * 0.25)
    budget = {
        NO_EXECUTION_FLAG: True,
        "api_calls": 0,
        "checkpoint_storage_gb": round(checkpoint_gb, 3),
        "cpu_hours": round(cpu_hours, 3),
        "disk_usage_gb": round(prediction_gb + checkpoint_gb + 2.0, 3),
        "eval_only_jobs": len(eval_only),
        "expected_predictions_file_gb": round(prediction_gb, 3),
        "gpu_hours": round(gpu_hours, 3),
        "jobs": len(jobs),
        "lora_training_jobs": 0,
        "notes": "Conservative planning estimate; not a guarantee.",
        "table_export_storage_gb": 0.1,
        "trainable_jobs": len(trainable),
    }
    output = resolve_path(output_path)
    ensure_dir(output.parent)
    write_json(output, budget)
    return budget


def _planned_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for experiment in manifest.get("experiments", []):
        for method in experiment.get("methods", []):
            for seed in experiment.get("seeds", []):
                rows.append({"dataset": experiment.get("dataset"), "method": str(method), "seed": int(seed)})
    return rows
