"""Failure audit helpers for pilot runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json


def audit_failures(run_dir: str | Path) -> dict[str, Any]:
    """Read method status artifacts and write a normalized failure report."""

    root = Path(run_dir)
    status_path = root / "method_status.csv"
    ablation_path = root / "ablation_results.csv"
    rows = _read_csv(status_path if status_path.is_file() else ablation_path)
    failures: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", row.get("method_status", "succeeded"))).lower()
        if status in {"succeeded", "success", "trained"}:
            continue
        method = row.get("method") or row.get("ablation") or "unknown"
        failures.append(
            {
                "category": _failure_category(status, str(row.get("message", ""))),
                "dataset": row.get("dataset", "unknown"),
                "message": row.get("message", row.get("failure", "")),
                "method": method,
                "status": status,
            }
        )
    categories: dict[str, int] = {}
    for failure in failures:
        categories[failure["category"]] = categories.get(failure["category"], 0) + 1
    report = {
        "blocks_paper_scale_readiness": bool(failures),
        "failure_categories": categories,
        "failure_count": len(failures),
        "failures": failures,
        "non_reportable_marker": "NON_REPORTABLE",
        "run_dir": str(root),
    }
    write_json(root / "failure_report.json", report)
    return report


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _failure_category(status: str, message: str) -> str:
    text = f"{status} {message}".lower()
    if "skip" in text or "unavailable" in text:
        return "skipped_dependency"
    if "validation" in text:
        return "validation_error"
    return "runtime_error"
