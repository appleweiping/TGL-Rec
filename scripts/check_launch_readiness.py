"""Validate Phase 8 launch package readiness without executing jobs."""

from __future__ import annotations

import argparse
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

    readiness_reports = sorted((launch_dir / "dataset_readiness").glob("*_readiness.json"))
    readiness = []
    for path in readiness_reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        readiness.append({"dataset": data.get("dataset"), "status": data.get("status"), "blocker": data.get("blocker")})
        if data.get("status") != "READY":
            warnings.append(f"dataset {data.get('dataset')} is {data.get('status')}")
    if not readiness_reports:
        errors.append("no dataset readiness reports found")

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


if __name__ == "__main__":
    raise SystemExit(main())
