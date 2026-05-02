"""Convert Amazon Reviews 2023 domains into unified multidomain JSONL artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from llm4rec.data.amazon_reviews_2023 import (
    choose_metadata_file,
    choose_review_file,
    inspect_amazon_reviews_2023,
    iter_records,
)
from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.data.schema_validation import normalize_interaction, normalize_item
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.utils.env import current_git_commit


class AmazonConversionError(ValueError):
    """Raised when conversion cannot proceed safely."""


def prepare_amazon_multidomain(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    materialize: bool = False,
    preflight: bool = False,
) -> dict[str, Any]:
    """Prepare sampled or full Amazon multidomain artifacts."""

    plan = _conversion_plan(config_path)
    if preflight:
        return preflight_amazon_multidomain(config_path)
    if not materialize and not dry_run:
        dry_run = True

    schema_report = inspect_amazon_reviews_2023(plan["config_path"], plan["schema_report_path"])
    progress = _ProgressLogger(plan["progress_log_path"])
    progress.log(f"start status={'materialize' if materialize else 'dry_run'} mode={plan['mode']}")
    if materialize:
        _assert_safe_outputs(plan)
        ensure_dir(plan["interactions_path"].parent)
        ensure_dir(plan["items_path"].parent)
        interactions_tmp = Path(str(plan["interactions_path"]) + ".tmp")
        items_tmp = Path(str(plan["items_path"]) + ".tmp")
        interactions_tmp.unlink(missing_ok=True)
        items_tmp.unlink(missing_ok=True)
        try:
            with interactions_tmp.open("w", encoding="utf-8", newline="\n") as interactions_handle, items_tmp.open(
                "w", encoding="utf-8", newline="\n"
            ) as items_handle:
                domain_reports = _convert_all_domains(plan, interactions_handle, items_handle, progress)
            interactions_tmp.replace(plan["interactions_path"])
            items_tmp.replace(plan["items_path"])
        except Exception:
            progress.log("conversion failed before atomic rename; final files left untouched")
            raise
    else:
        domain_reports = _convert_all_domains(plan, None, None, progress)
    report = _conversion_report(plan, domain_reports, schema_report, dry_run=dry_run, materialized=materialize)
    write_json(plan["report_path"], report)
    write_json(plan["domain_counts_path"], {"domains": domain_reports, "totals": report["summary"]})
    progress.log(f"completed status={report['status']}")
    return report


def preflight_amazon_multidomain(config_path: str | Path) -> dict[str, Any]:
    """Write a full/sampled conversion preflight without converting rows."""

    plan = _conversion_plan(config_path)
    schema_report = inspect_amazon_reviews_2023(plan["config_path"], plan["schema_report_path"])
    domains = {}
    estimated_reviews = 0
    estimated_items = 0
    for domain, raw_path in plan["raw_domains"].items():
        inspected = schema_report["domains"].get(domain, {})
        estimates = inspected.get("row_count_estimate", {})
        estimated_reviews += int(estimates.get("reviews") or 0)
        estimated_items += int(estimates.get("items") or 0)
        domains[domain] = {
            "input_metadata_file": inspected.get("metadata_file_candidate"),
            "input_review_file": inspected.get("review_file_candidate"),
            "estimated_item_rows": estimates.get("items"),
            "estimated_review_rows": estimates.get("reviews"),
            "raw_path": str(resolve_path(raw_path)),
            "status": inspected.get("status"),
        }
    disk = _disk_report(plan["output_dir"])
    preflight = {
        NO_EXECUTION_FLAG: True,
        "available_disk": disk,
        "config_hash": _file_sha256(plan["config_path"]),
        "conversion_mode": plan["mode"],
        "code_commit": current_git_commit(resolve_path(".")),
        "domains": domains,
        "estimated_input_item_rows": estimated_items,
        "estimated_input_review_rows": estimated_reviews,
        "estimated_output_size_bytes": _estimated_output_size(estimated_reviews, estimated_items),
        "expected_output_paths": {
            "conversion_report": str(plan["report_path"]),
            "domain_counts": str(plan["domain_counts_path"]),
            "interactions": str(plan["interactions_path"]),
            "items": str(plan["items_path"]),
            "schema_report": str(plan["schema_report_path"]),
        },
        "existing_outputs": {
            "conversion_report": plan["report_path"].exists(),
            "domain_counts": plan["domain_counts_path"].exists(),
            "interactions": plan["interactions_path"].exists(),
            "items": plan["items_path"].exists(),
            "schema_report": plan["schema_report_path"].exists(),
        },
        "is_experiment": False,
        "not_an_experiment_warning": "This is data preparation only: no training, evaluation, API, LoRA, or paper results.",
        "overwrite_enabled": plan["overwrite_existing"],
        "resume_enabled": plan["resume"],
        "start_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(plan["preflight_path"], preflight)
    return preflight


def _convert_all_domains(
    plan: dict[str, Any],
    interactions_handle: TextIO | None,
    items_handle: TextIO | None,
    progress: "_ProgressLogger",
) -> dict[str, dict[str, Any]]:
    domain_reports: dict[str, dict[str, Any]] = {}
    for domain, raw_path in plan["raw_domains"].items():
        progress.log(f"domain {domain}: start")
        domain_reports[domain] = _convert_domain(
            str(domain),
            resolve_path(str(raw_path)),
            max_interactions=plan["max_interactions_per_domain"],
            max_items=plan["max_items_per_domain"],
            interactions_handle=interactions_handle,
            items_handle=items_handle,
            drop_missing_text_items=plan["drop_missing_text_items"],
            progress=progress,
        )
        progress.log(f"domain {domain}: done {json.dumps(domain_reports[domain], sort_keys=True)}")
    return domain_reports


def _convert_domain(
    domain: str,
    directory: Path,
    *,
    max_interactions: int | None,
    max_items: int | None,
    interactions_handle: TextIO | None,
    items_handle: TextIO | None,
    drop_missing_text_items: bool,
    progress: "_ProgressLogger",
) -> dict[str, Any]:
    files = sorted([file for file in directory.iterdir() if file.is_file()], key=lambda value: value.name) if directory.is_dir() else []
    review_file = choose_review_file(files)
    metadata_file = choose_metadata_file(files)
    report: dict[str, Any] = {
        "detected_metadata_file": str(metadata_file) if metadata_file else None,
        "detected_review_file": str(review_file) if review_file else None,
        "domain": domain,
        "dropped_interactions": 0,
        "drop_reasons": {},
        "duplicate_interactions": 0,
        "duplicate_items": 0,
        "items_written": 0,
        "metadata_coverage": 0.0,
        "metadata_missing": metadata_file is None,
        "missing_text_items": 0,
        "raw_interactions": 0,
        "raw_items": 0,
        "status": "READY",
        "unique_interaction_items": 0,
        "valid_interactions": 0,
        "valid_items": 0,
        "warnings": [],
    }
    if review_file is None:
        report["status"] = "MISSING_REVIEWS"
        report["warnings"].append("review/interactions file missing")
        return report

    seen_interactions: set[tuple[str, str, int, str]] = set()
    interaction_item_ids: set[str] = set()
    drop_reasons: Counter[str] = Counter()
    for raw in iter_records(review_file):
        report["raw_interactions"] += 1
        row, reason = normalize_interaction(raw, domain)
        if row is None:
            drop_reasons[str(reason)] += 1
            if report["raw_interactions"] % 500000 == 0:
                progress.log(f"domain {domain}: read {report['raw_interactions']} reviews")
            continue
        key = (row["user_id"], row["item_id"], int(row["timestamp"]), row["domain"])
        if key in seen_interactions:
            report["duplicate_interactions"] += 1
            continue
        seen_interactions.add(key)
        interaction_item_ids.add(row["item_id"])
        report["valid_interactions"] += 1
        if interactions_handle is not None:
            _write_json_line(interactions_handle, row)
        if max_interactions is not None and report["valid_interactions"] >= max_interactions:
            break
        if report["raw_interactions"] % 500000 == 0:
            progress.log(f"domain {domain}: read {report['raw_interactions']} reviews")
    report["dropped_interactions"] = sum(drop_reasons.values())
    report["drop_reasons"] = dict(sorted(drop_reasons.items()))
    report["unique_interaction_items"] = len(interaction_item_ids)

    if metadata_file is None:
        report["status"] = "PARTIAL"
        report["warnings"].append("metadata/items file missing")
        return report

    seen_items: set[tuple[str, str]] = set()
    remaining_item_ids = set(interaction_item_ids)
    for raw in iter_records(metadata_file):
        report["raw_items"] += 1
        item, reason = normalize_item(raw, domain)
        if item is None:
            continue
        key = (item["item_id"], item["domain"])
        if key in seen_items:
            report["duplicate_items"] += 1
            continue
        if interaction_item_ids and item["item_id"] not in interaction_item_ids:
            continue
        seen_items.add(key)
        report["valid_items"] += 1
        if reason == "missing_text":
            report["missing_text_items"] += 1
            if drop_missing_text_items:
                continue
        if items_handle is not None:
            _write_json_line(items_handle, item)
        report["items_written"] += 1
        remaining_item_ids.discard(item["item_id"])
        if max_items is not None and report["items_written"] >= max_items:
            break
        if interaction_item_ids and not remaining_item_ids:
            break
        if report["raw_items"] % 250000 == 0:
            progress.log(f"domain {domain}: read {report['raw_items']} metadata rows")
    missing_metadata = len(interaction_item_ids) - report["valid_items"]
    report["metadata_coverage"] = 0.0 if not interaction_item_ids else report["valid_items"] / len(interaction_item_ids)
    if missing_metadata > 0:
        report["warnings"].append(f"{missing_metadata} interacted items missing metadata")
        report["status"] = "PARTIAL"
    return report


def _conversion_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    dataset = dict(config.get("dataset", config))
    conversion = dict(config.get("conversion", dataset.get("conversion", {})))
    source_config_path = dataset.get("source_config", config.get("source_config", "configs/datasets/amazon_reviews_2023.yaml"))
    source_config = load_yaml_config(source_config_path)
    raw_domains = dict(source_config.get("dataset", {}).get("raw_domains", source_config.get("raw_domains", {})))
    if not raw_domains:
        raise ValueError("Amazon multidomain conversion requires raw_domains in source config")
    paths = dict(dataset.get("paths", {}))
    interactions_path = resolve_path(paths.get("interactions") or dataset.get("interactions_path") or "data/raw/amazon_multidomain/interactions.jsonl")
    items_path = resolve_path(paths.get("items") or dataset.get("items_path") or "data/raw/amazon_multidomain/items.jsonl")
    output_dir = ensure_dir(interactions_path.parent)
    report_path = _path_value(paths.get("conversion_report") or dataset.get("conversion_report_path"), output_dir / "conversion_report.json")
    preflight_path = _path_value(paths.get("conversion_preflight") or dataset.get("conversion_preflight_path"), output_dir / "conversion_preflight.json")
    domain_counts_path = _path_value(paths.get("domain_counts") or dataset.get("domain_counts_path"), output_dir / "domain_counts.json")
    schema_report_path = _path_value(paths.get("schema_report") or dataset.get("schema_report_path"), output_dir / "schema_report.json")
    progress_log_path = _path_value(paths.get("progress_log") or dataset.get("progress_log_path"), output_dir / "conversion_progress.log")
    sample = dict(dataset.get("sample", {}))
    max_interactions_per_domain = _optional_int(sample.get("max_interactions_per_domain"))
    max_items_per_domain = _optional_int(sample.get("max_items_per_domain"))
    return {
        "config": config,
        "config_path": resolve_path(config_path),
        "domain_counts_path": domain_counts_path,
        "drop_missing_text_items": bool(conversion.get("drop_missing_text_items", False)),
        "interactions_path": interactions_path,
        "items_path": items_path,
        "max_interactions_per_domain": max_interactions_per_domain,
        "max_items_per_domain": max_items_per_domain,
        "mode": "sampled" if max_interactions_per_domain or max_items_per_domain else "full",
        "output_dir": output_dir,
        "overwrite_existing": bool(conversion.get("overwrite_existing", False)),
        "preflight_path": preflight_path,
        "progress_log_path": progress_log_path,
        "raw_domains": {str(domain): str(path) for domain, path in raw_domains.items()},
        "report_path": report_path,
        "resume": bool(conversion.get("resume", True)),
        "schema_report_path": schema_report_path,
        "source_config_path": resolve_path(source_config_path),
    }


def _conversion_report(
    plan: dict[str, Any],
    domain_reports: dict[str, dict[str, Any]],
    schema_report: dict[str, Any],
    *,
    dry_run: bool,
    materialized: bool,
) -> dict[str, Any]:
    summary = _summary(domain_reports)
    report = {
        NO_EXECUTION_FLAG: True,
        "code_commit": current_git_commit(resolve_path(".")),
        "config_hash": _file_sha256(plan["config_path"]),
        "domains": domain_reports,
        "dry_run": bool(dry_run),
        "interactions_path": str(plan["interactions_path"]),
        "items_path": str(plan["items_path"]),
        "materialized": bool(materialized),
        "mode": plan["mode"],
        "output_paths": {
            "conversion_preflight": str(plan["preflight_path"]),
            "conversion_report": str(plan["report_path"]),
            "domain_counts": str(plan["domain_counts_path"]),
            "interactions": str(plan["interactions_path"]),
            "items": str(plan["items_path"]),
            "schema_report": str(plan["schema_report_path"]),
        },
        "schema_report_path": str(plan["schema_report_path"]),
        "source_config": str(plan["source_config_path"]),
        "status": "DRY_RUN" if dry_run and not materialized else "MATERIALIZED",
        "summary": summary,
        "warnings": _warnings(domain_reports, schema_report),
    }
    return report


def _summary(domain_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unique_items = sum(int(row.get("unique_interaction_items", 0)) for row in domain_reports.values())
    valid_items = sum(int(row.get("valid_items", 0)) for row in domain_reports.values())
    dropped_interactions = sum(int(row.get("dropped_interactions", 0)) for row in domain_reports.values())
    duplicate_interactions = sum(int(row.get("duplicate_interactions", 0)) for row in domain_reports.values())
    duplicate_items = sum(int(row.get("duplicate_items", 0)) for row in domain_reports.values())
    missing_text_items = sum(int(row.get("missing_text_items", 0)) for row in domain_reports.values())
    return {
        "dropped_interactions": dropped_interactions,
        "dropped_rows": dropped_interactions,
        "duplicate_interactions": duplicate_interactions,
        "duplicate_interactions_removed": duplicate_interactions,
        "duplicate_items": duplicate_items,
        "duplicate_items_removed": duplicate_items,
        "items_written": sum(int(row.get("items_written", 0)) for row in domain_reports.values()),
        "metadata_coverage": 0.0 if unique_items == 0 else valid_items / unique_items,
        "items_missing_text": missing_text_items,
        "missing_text_items": missing_text_items,
        "raw_interactions": sum(int(row.get("raw_interactions", 0)) for row in domain_reports.values()),
        "raw_items": sum(int(row.get("raw_items", 0)) for row in domain_reports.values()),
        "unique_interaction_items": unique_items,
        "valid_interactions": sum(int(row.get("valid_interactions", 0)) for row in domain_reports.values()),
        "valid_items": valid_items,
    }


def _warnings(domain_reports: dict[str, dict[str, Any]], schema_report: dict[str, Any]) -> list[str]:
    warnings = []
    if schema_report.get("overall_status") != "convertible":
        warnings.append("schema inspection reported partial convertibility")
    for domain, row in domain_reports.items():
        for warning in row.get("warnings", []):
            warnings.append(f"{domain}: {warning}")
    return warnings


def _assert_safe_outputs(plan: dict[str, Any]) -> None:
    final_paths = [plan["interactions_path"], plan["items_path"]]
    existing = [str(path) for path in final_paths if path.exists()]
    if existing and not plan["overwrite_existing"]:
        raise AmazonConversionError(
            "Refusing to overwrite existing full converted files without overwrite_existing=true: "
            + ", ".join(existing)
        )
    if existing and plan["overwrite_existing"]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in final_paths:
            if path.exists():
                path.replace(Path(str(path) + f".bak.{timestamp}"))


def _write_json_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _disk_report(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
    }


def _estimated_output_size(estimated_reviews: int, estimated_items: int) -> int:
    return int(estimated_reviews * 120 + estimated_items * 280)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _path_value(value: Any, default: Path) -> Path:
    return resolve_path(value) if value else default


class _ProgressLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{timestamp}\t{message}\n")
