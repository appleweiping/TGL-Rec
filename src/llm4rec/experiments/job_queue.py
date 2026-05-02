"""Paper-scale job queue generation without execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, write_csv_rows, write_json, write_jsonl


def create_job_queue(
    manifest: dict[str, Any],
    output_dir: str | Path = "outputs/launch/paper_v1",
) -> list[dict[str, Any]]:
    """Write planned paper-scale jobs from a launch manifest."""

    output = ensure_dir(resolve_path(output_dir))
    jobs: list[dict[str, Any]] = []
    counter = 1
    for experiment in manifest.get("experiments", []):
        for method in experiment.get("methods", []):
            for seed in experiment.get("seeds", []):
                job_id = f"paper_v1_{counter:06d}"
                run_dir = Path(str(experiment.get("output_dir"))) / str(experiment.get("dataset")) / str(method) / f"seed_{seed}"
                jobs.append(
                    {
                        NO_EXECUTION_FLAG: True,
                        "allow_api_calls": False,
                        "command": (
                            f"python scripts/run_experiment.py --config {experiment['config_path']} "
                            f"--method {method} --seed {seed}"
                        ),
                        "config_path": experiment["config_path"],
                        "dataset": experiment.get("dataset"),
                        "dependencies": json.dumps([f"protocol:{manifest.get('protocol_version')}"]),
                        "estimated_memory": _memory_estimate(method),
                        "estimated_runtime": _runtime_estimate(method),
                        "job_id": job_id,
                        "method": method,
                        "output_dir": str(run_dir),
                        "protocol_version": manifest.get("protocol_version"),
                        "reportable": True,
                        "seed": int(seed),
                        "status": "planned",
                    }
                )
                counter += 1
    write_jsonl(output / "jobs.jsonl", jobs)
    write_csv_rows(output / "jobs.csv", jobs)
    write_json(output / "launch_manifest.json", manifest)
    write_json(output / "go_no_go_checklist.json", _checklist_data(manifest, jobs))
    (output / "go_no_go_checklist.md").write_text(_checklist_markdown(manifest, jobs), encoding="utf-8", newline="\n")
    return jobs


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _runtime_estimate(method: str) -> str:
    trainable = {"mf", "bpr", "sasrec", "temporal_graph_encoder", "time_graph_evidence_dynamic"}
    return "4h" if method in trainable else "1h"


def _memory_estimate(method: str) -> str:
    if method in {"sasrec", "temporal_graph_encoder", "time_graph_evidence_dynamic"}:
        return "8GB"
    return "4GB"


def _checklist_data(manifest: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        NO_EXECUTION_FLAG: True,
        "items": {
            "api_calls_planned_zero": manifest.get("api_calls_planned") == 0,
            "jobs_all_planned": all(job.get("status") == "planned" for job in jobs),
            "lora_jobs_planned_zero": manifest.get("lora_training_jobs_planned") == 0,
            "protocol_version_present": bool(manifest.get("protocol_version")),
            "resource_budget_planned": True,
            "table_plan_planned": True,
        },
        "status": "PLANNED_REQUIRES_DATA_AND_USER_CONFIRMATION",
    }


def _checklist_markdown(manifest: dict[str, Any], jobs: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 8 Go/No-Go Checklist",
        "",
        f"NO_EXPERIMENTS_EXECUTED_IN_PHASE_8 = {str(True).lower()}",
        "",
        "- [ ] Full datasets are READY in dataset_readiness outputs.",
        "- [x] Protocol version is declared.",
        "- [x] Paper configs are planned and reportable-safe.",
        "- [x] Job queue is generated with status=planned.",
        "- [x] API calls planned: 0.",
        "- [x] LoRA training jobs planned: 0.",
        "- [ ] User explicitly confirms launch.",
        "",
        f"Protocol version: {manifest.get('protocol_version')}",
        f"Planned jobs: {len(jobs)}",
    ]
    return "\n".join(lines) + "\n"
