"""Validate Phase 8 launch package readiness without executing jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.readiness import NO_EXECUTION_FLAG  # noqa: E402
from llm4rec.experiments.config import resolve_path  # noqa: E402
from llm4rec.experiments.job_queue import load_manifest  # noqa: E402
from llm4rec.experiments.validate import validate_experiment_config  # noqa: E402
from llm4rec.io.artifacts import read_jsonl, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    report = check_launch_readiness(args.manifest)
    print(json.dumps({"status": report["status"], "validation_errors": len(report["errors"])}, sort_keys=True))
    return 0


def check_launch_readiness(manifest_path: str | Path) -> dict[str, object]:
    manifest_file = resolve_path(manifest_path)
    manifest = load_manifest(manifest_file)
    launch_dir = manifest_file.parent
    validation_dir = launch_dir / "validation"
    errors: list[str] = []
    warnings: list[str] = []

    for rel in [
        "protocol/protocol_manifest.json",
        "protocol/frozen_split_manifest.json",
        "protocol/frozen_candidate_manifest.json",
        "jobs.jsonl",
        "resource_budget.json",
        "table_plan.json",
        "go_no_go_checklist.md",
    ]:
        if not (launch_dir / rel).is_file():
            errors.append(f"missing launch artifact: {rel}")
    errors.extend(_materialized_artifact_errors(launch_dir))

    readiness_reports = sorted((launch_dir / "dataset_readiness").glob("*_readiness.json"))
    manifest_datasets = {str(dataset) for dataset in manifest.get("datasets", [])}
    readiness = []
    extra_readiness = []
    seen_manifest_datasets: set[str] = set()
    for path in readiness_reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"dataset": data.get("dataset"), "status": data.get("status"), "blocker": data.get("blocker")}
        if str(data.get("dataset")) in manifest_datasets:
            readiness.append(row)
            seen_manifest_datasets.add(str(data.get("dataset")))
            if data.get("status") != "READY":
                warnings.append(f"dataset {data.get('dataset')} is {data.get('status')}")
        else:
            extra_readiness.append(row)
    if not readiness_reports:
        errors.append("no dataset readiness reports found")
    for dataset in sorted(manifest_datasets - seen_manifest_datasets):
        errors.append(f"dataset readiness report missing for manifest dataset: {dataset}")

    jobs = read_jsonl(launch_dir / "jobs.jsonl") if (launch_dir / "jobs.jsonl").is_file() else []
    executed_jobs = [job for job in jobs if job.get("status") != "planned"]
    if executed_jobs:
        errors.append("job queue contains non-planned jobs")
    for job in jobs:
        output_dir = resolve_path(str(job.get("output_dir")))
        if (output_dir / "metrics.json").exists() or (output_dir / "predictions.jsonl").exists():
            errors.append(f"planned output dir already contains result artifacts: {output_dir}")
            break
        if job.get("allow_api_calls") is not False:
            errors.append(f"job allows API calls: {job.get('job_id')}")
            break

    for config_path in manifest.get("experiment_configs", []):
        try:
            validate_experiment_config(config_path)
        except Exception as exc:  # noqa: BLE001 - collect all validation failures.
            errors.append(f"{config_path}: {exc}")

    if errors:
        status = "NO_GO"
    elif any(row["status"] != "READY" for row in readiness):
        status = "CONDITIONAL_GO"
    else:
        status = "GO"

    report = {
        NO_EXECUTION_FLAG: True,
        "dataset_readiness": readiness,
        "errors": errors,
        "extra_readiness_reports": extra_readiness,
        "jobs_planned": len(jobs),
        "manifest": str(manifest_file),
        "status": status,
        "warnings": warnings,
    }
    write_json(validation_dir / "launch_readiness.json", report)
    write_json(
        validation_dir / "validation_report.json",
        {
            NO_EXECUTION_FLAG: True,
            "errors": errors,
            "status": status,
            "validated_configs": manifest.get("experiment_configs", []),
            "warnings": warnings,
        },
    )
    return report


def _materialized_artifact_errors(launch_dir: Path) -> list[str]:
    errors: list[str] = []
    for rel, key in [
        ("protocol/frozen_split_manifest.json", "split_artifacts"),
        ("protocol/frozen_candidate_manifest.json", "candidate_artifacts"),
    ]:
        path = launch_dir / rel
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not bool(data.get("materialized")):
            continue
        entries = data.get(key) or data.get("planned_" + key) or []
        for entry in entries:
            artifact_path = resolve_path(str(entry.get("path", "")))
            if not artifact_path.is_file():
                errors.append(f"materialized artifact missing: {artifact_path}")
                continue
            expected_sha = entry.get("sha256")
            if expected_sha and _sha256(artifact_path) != expected_sha:
                errors.append(f"materialized artifact checksum mismatch: {artifact_path}")
            expected_rows = entry.get("rows")
            if expected_rows is not None and int(expected_rows) <= 0:
                errors.append(f"materialized artifact has no rows: {artifact_path}")
            if entry.get("artifact_type") == "candidate":
                if entry.get("target_included_rows") != entry.get("rows"):
                    errors.append(f"candidate artifact has rows without target item: {artifact_path}")
                pool = entry.get("candidate_pool_artifact")
                if isinstance(pool, dict) and pool.get("path"):
                    pool_path = resolve_path(str(pool.get("path")))
                    if not pool_path.is_file():
                        errors.append(f"candidate pool artifact missing: {pool_path}")
                    elif pool.get("sha256") and _sha256(pool_path) != pool.get("sha256"):
                        errors.append(f"candidate pool checksum mismatch: {pool_path}")
    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
