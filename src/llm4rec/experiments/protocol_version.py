"""Protocol version freeze utilities for paper launch preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.experiments.manifest import manifest_from_config
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.utils.env import current_git_commit

DEFAULT_PAPER_CONFIGS = [
    "configs/experiments/paper_movielens_accuracy.yaml",
    "configs/experiments/paper_movielens_ablation.yaml",
    "configs/experiments/paper_movielens_long_tail.yaml",
    "configs/experiments/paper_movielens_efficiency.yaml",
    "configs/experiments/paper_amazon_multidomain_accuracy.yaml",
    "configs/experiments/paper_amazon_multidomain_ablation.yaml",
    "configs/experiments/paper_amazon_multidomain_cold_start.yaml",
    "configs/experiments/paper_amazon_multidomain_efficiency.yaml",
]

DEFAULT_DATASET_CONFIGS = [
    "configs/datasets/movielens_full.yaml",
    "configs/datasets/amazon_multidomain_filtered_iterative_k3.yaml",
]


class ProtocolFreezeError(ValueError):
    """Raised when a protocol version cannot be safely written."""


def freeze_protocol(
    version: str,
    output_dir: str | Path = "outputs/launch/paper_v1/protocol",
    *,
    config_paths: list[str | Path] | None = None,
    dataset_config_paths: list[str | Path] | None = None,
    dry_run: bool = True,
    materialize: bool = False,
    force_new_version: bool = False,
) -> dict[str, Any]:
    """Freeze the experiment protocol metadata without running experiments."""

    output = ensure_dir(resolve_path(output_dir))
    manifest_path = output / "protocol_manifest.json"
    preserved = _preserve_existing_materialized_protocol(output, version, dry_run=dry_run)
    if manifest_path.exists() and not dry_run and not force_new_version:
        raise ProtocolFreezeError(
            f"Refusing to overwrite existing protocol manifest without --force-new-version: {manifest_path}"
        )
    config_paths = config_paths or DEFAULT_PAPER_CONFIGS
    dataset_config_paths = dataset_config_paths or DEFAULT_DATASET_CONFIGS
    experiments = [_experiment_entry(path) for path in config_paths]
    methods = sorted({method for entry in experiments for method in entry["methods"]})
    metrics = sorted({metric for entry in experiments for metric in entry["metrics"]})
    seeds = sorted({seed for entry in experiments for seed in entry["seeds"]})
    dataset_hashes = [
        {"config_path": str(resolve_path(path)), "sha256": _file_sha256(resolve_path(path))}
        for path in dataset_config_paths
    ]
    manifest = {
        NO_EXECUTION_FLAG: True,
        "candidate_protocol": "sampled_fixed_for_pilot_full_or_fixed_for_paper",
        "code_commit": current_git_commit(resolve_path(".")),
        "dataset_config_hashes": dataset_hashes,
        "dry_run": bool(dry_run),
        "experiment_config_hashes": [
            {"config_path": entry["config_path"], "sha256": _file_sha256(Path(entry["config_path"]))}
            for entry in experiments
        ],
        "experiments": experiments,
        "materialized": bool(materialize) and not dry_run,
        "metric_config": metrics,
        "method_set": methods,
        "protocol_version": version,
        "seed_set": seeds,
        "split_protocol": "leave_one_out",
        "status": "DRY_RUN_ONLY" if dry_run else "FROZEN_METADATA",
    }
    if preserved is not None:
        return {**manifest, "preserved_existing_materialization": True}
    split_manifest = {
        NO_EXECUTION_FLAG: True,
        "materialized": bool(materialize) and not dry_run,
        "planned_split_artifacts": _artifact_entries(experiments, "split_artifact"),
        "protocol_version": version,
        "split_protocol": "leave_one_out",
    }
    candidate_manifest = {
        NO_EXECUTION_FLAG: True,
        "candidate_protocol": "fixed_shared_candidates",
        "materialized": bool(materialize) and not dry_run,
        "planned_candidate_artifacts": _artifact_entries(experiments, "candidate_artifact"),
        "protocol_version": version,
    }
    write_json(manifest_path, manifest)
    write_json(output / "frozen_split_manifest.json", split_manifest)
    write_json(output / "frozen_candidate_manifest.json", candidate_manifest)
    return manifest


def _preserve_existing_materialized_protocol(
    output: Path,
    version: str,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Avoid downgrading materialized Phase 9 manifests during dry-run checks."""

    if not dry_run:
        return None
    manifest_path = output / "protocol_manifest.json"
    split_path = output / "frozen_split_manifest.json"
    candidate_path = output / "frozen_candidate_manifest.json"
    if not (manifest_path.is_file() and split_path.is_file() and candidate_path.is_file()):
        return None
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        str(current.get("protocol_version")) == str(version)
        and bool(current.get("materialized"))
        and bool(split.get("materialized"))
        and bool(candidate.get("materialized"))
    ):
        return current
    return None


def _experiment_entry(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    config = load_yaml_config(resolved)
    manifest = manifest_from_config(config).data
    protocol = config.get("protocol_version", manifest.get("protocol_version"))
    return {
        "candidate_artifact": config.get("candidate_artifact") or config.get("artifacts", {}).get("candidate_artifact"),
        "candidate_strategy": manifest.get("candidate_strategy"),
        "config_path": str(resolved),
        "dataset": manifest.get("dataset"),
        "metrics": [str(metric) for metric in manifest.get("metrics", [])],
        "methods": [str(method.get("name", method)) if isinstance(method, dict) else str(method) for method in manifest.get("methods", [])],
        "output_dir": manifest.get("output_dir"),
        "protocol_version": protocol,
        "reportable": bool(manifest.get("reportable")),
        "run_mode": manifest.get("run_mode"),
        "seeds": [int(seed) for seed in manifest.get("seeds", [])],
        "split_artifact": config.get("split_artifact") or config.get("artifacts", {}).get("split_artifact"),
        "split_strategy": manifest.get("split_strategy"),
    }


def _artifact_entries(experiments: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in experiments:
        value = str(entry.get(key) or "")
        if not value:
            continue
        seen[(str(entry.get("dataset")), value)] = {
            "dataset": entry.get("dataset"),
            "path": value,
            "status": "planned",
        }
    return list(seen.values())


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
