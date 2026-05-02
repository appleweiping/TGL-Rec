"""Metadata-only data artifact freeze planning for Phase 8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json


def plan_data_artifact_freeze(
    dataset_config_paths: list[str | Path],
    output_dir: str | Path = "outputs/launch/paper_v1/protocol",
    *,
    materialize: bool = False,
) -> dict[str, Any]:
    """Create metadata-only split/candidate freeze manifests.

    ``materialize=False`` is the Phase 8 default. It records planned artifact paths
    and config provenance without constructing paper-scale splits or candidates.
    """

    output = ensure_dir(resolve_path(output_dir))
    datasets = []
    for config_path in dataset_config_paths:
        config = load_yaml_config(config_path)
        dataset = dict(config.get("dataset", config))
        name = str(dataset.get("name", Path(config_path).stem))
        datasets.append(
            {
                "candidate_artifact": str(resolve_path(f"outputs/artifacts/protocol_v1/{name}/candidates.jsonl")),
                "config_path": str(resolve_path(config_path)),
                "dataset": name,
                "materialized": False,
                "split_artifact": str(resolve_path(f"outputs/artifacts/protocol_v1/{name}/splits.jsonl")),
            }
        )
    manifest = {
        NO_EXECUTION_FLAG: True,
        "datasets": datasets,
        "materialize_requested": bool(materialize),
        "status": "planned_only" if not materialize else "materialize_requested_not_run_by_phase8",
    }
    write_json(output / "data_artifact_freeze_plan.json", manifest)
    return manifest
