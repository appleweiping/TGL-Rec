"""Dataset readiness checks for paper-scale launch preparation."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from llm4rec.data.movielens_adapter import load_movielens_style
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import read_jsonl, write_json

NO_EXECUTION_FLAG = "NO_EXPERIMENTS_EXECUTED_IN_PHASE_8"


class DatasetReadinessError(ValueError):
    """Raised for invalid readiness configuration, not for missing data."""


def check_dataset_readiness(config_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Validate dataset availability and schema without materializing experiments.

    Missing datasets are reported as structured ``MISSING`` readiness output instead of
    crashing, so launch checks can distinguish absent data from broken code.
    """

    config = load_yaml_config(config_path)
    dataset = dict(config.get("dataset", config))
    name = str(dataset.get("name", Path(config_path).stem))
    readiness = dict(config.get("readiness", {}))
    output = Path(output_path or readiness.get("output_path") or _default_output_path(name))

    if bool(readiness.get("allow_download", False)):
        raise DatasetReadinessError("Phase 8 readiness does not download data automatically")

    try:
        interactions, items = _load_dataset_tables(dataset)
    except FileNotFoundError as exc:
        report = _missing_report(name, dataset, str(exc))
        write_json(output, report)
        return report

    report = _validate_tables(name, dataset, interactions, items)
    write_json(output, report)
    return report


def _load_dataset_tables(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = str(dataset.get("adapter", "generic_jsonl"))
    paths = dict(dataset.get("paths", {}))
    if adapter == "movielens_style":
        return load_movielens_style(paths)
    interactions_path = paths.get("interactions")
    items_path = paths.get("items")
    if not interactions_path or not items_path:
        raise FileNotFoundError("Dataset config must provide paths.interactions and paths.items")
    interactions_file = resolve_path(interactions_path)
    items_file = resolve_path(items_path)
    if not interactions_file.is_file() or not items_file.is_file():
        raise FileNotFoundError(
            f"Missing dataset files: interactions={interactions_file}, items={items_file}"
        )
    return _read_table(interactions_file), _read_table(items_file)


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise FileNotFoundError(f"Unsupported readiness table format: {path}")


def _validate_tables(
    name: str,
    dataset: dict[str, Any],
    interactions: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    min_user_interactions = int(dataset.get("min_user_interactions", 3))
    min_item_interactions = int(dataset.get("min_item_interactions", 1))
    interaction_fields = set().union(*(row.keys() for row in interactions)) if interactions else set()
    item_fields = set().union(*(row.keys() for row in items)) if items else set()
    required_interactions = _required_list(
        dataset.get("required_interaction_columns"),
        ["user_id", "item_id", "timestamp", "rating_or_interaction"],
    )
    required_items = _required_list(dataset.get("required_item_columns"), ["item_id", "text_field"])

    missing_fields: list[str] = []
    for field in required_interactions:
        if field == "rating_or_interaction":
            if not ({"rating", "interaction", "interaction_value", "value"} & interaction_fields):
                missing_fields.append("rating_or_interaction")
        elif field not in interaction_fields:
            missing_fields.append(field)
    for field in required_items:
        if field == "text_field":
            if not ({"title", "raw_text", "description"} & item_fields):
                missing_fields.append("item_text_field")
        elif field not in item_fields:
            missing_fields.append(field)

    user_counts = Counter(str(row.get("user_id")) for row in interactions if row.get("user_id") not in (None, ""))
    use_domain_keys = "domain" in interaction_fields or "domain" in item_fields
    item_ids = {
        _domain_item_key(row) if use_domain_keys else str(row.get("item_id"))
        for row in interactions
        if row.get("item_id") not in (None, "")
    }
    metadata_item_ids = {
        _domain_item_key(row) if use_domain_keys else str(row.get("item_id"))
        for row in items
        if row.get("item_id") not in (None, "")
    }
    item_counts = Counter(
        _domain_item_key(row) if use_domain_keys else str(row.get("item_id"))
        for row in interactions
        if row.get("item_id") not in (None, "")
    )
    timestamps = [_to_float(row.get("timestamp")) for row in interactions if _to_float(row.get("timestamp")) is not None]
    missing_timestamps = sum(1 for row in interactions if row.get("timestamp") in (None, ""))
    duplicate_count = _duplicate_interactions(interactions)
    users_too_few = sum(1 for count in user_counts.values() if count < min_user_interactions)
    items_too_few = sum(1 for count in item_counts.values() if count < min_item_interactions)
    items_without_text = sum(1 for row in items if not _has_item_text(row))
    domains = sorted({str(row.get("domain")) for row in interactions + items if row.get("domain") not in (None, "")})

    issues: list[str] = []
    if not interactions:
        issues.append("no interactions loaded")
    if not items:
        issues.append("no item metadata loaded")
    if missing_fields:
        issues.append("required fields missing")
    if missing_timestamps:
        issues.append("missing timestamps detected")
    if duplicate_count:
        issues.append("duplicate interactions detected")
    if users_too_few:
        issues.append("users with too few interactions detected")
    if items_too_few:
        issues.append("items with too few interactions detected")
    if item_ids - metadata_item_ids:
        issues.append("some interacted items have no item metadata")
    if items_without_text:
        issues.append("items without title/raw_text/description detected")

    status = "READY" if not issues else "PARTIAL"
    blocker = status != "READY"
    user_count = len(user_counts)
    item_count = len(metadata_item_ids)
    possible = user_count * item_count
    sparsity = None if possible == 0 else 1.0 - (len(interactions) / possible)
    return {
        NO_EXECUTION_FLAG: True,
        "blocker": blocker,
        "counts": {
            "domains": len(domains),
            "interactions": len(interactions),
            "items": item_count,
            "users": user_count,
        },
        "dataset": name,
        "domains": domains,
        "duplicate_interactions": duplicate_count,
        "items_without_text_fields": items_without_text,
        "items_with_too_few_interactions": items_too_few,
        "leave_one_out_feasible": users_too_few == 0 and missing_timestamps == 0 and bool(interactions),
        "missing_fields": missing_fields,
        "missing_timestamps": missing_timestamps,
        "paths": dict(dataset.get("paths", {})),
        "sparsity": sparsity,
        "status": status,
        "timestamp_max": max(timestamps) if timestamps else None,
        "timestamp_min": min(timestamps) if timestamps else None,
        "users_with_too_few_interactions": users_too_few,
        "issues": issues,
    }


def _missing_report(name: str, dataset: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        NO_EXECUTION_FLAG: True,
        "blocker": True,
        "counts": {"domains": 0, "interactions": 0, "items": 0, "users": 0},
        "dataset": name,
        "domains": [],
        "duplicate_interactions": 0,
        "items_without_text_fields": 0,
        "missing_fields": [],
        "missing_timestamps": 0,
        "paths": dict(dataset.get("paths", {})),
        "sparsity": None,
        "status": "MISSING",
        "timestamp_max": None,
        "timestamp_min": None,
        "users_with_too_few_interactions": 0,
        "issues": [message],
    }


def _duplicate_interactions(interactions: list[dict[str, Any]]) -> int:
    use_domain = any("domain" in row for row in interactions)
    counts = Counter(
        (
            str(row.get("user_id")),
            str(row.get("item_id")),
            str(row.get("timestamp")),
            str(row.get("domain")) if use_domain else "",
        )
        for row in interactions
    )
    return sum(count - 1 for count in counts.values() if count > 1)


def _domain_item_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("domain") or ""), str(row.get("item_id")))


def _has_item_text(row: dict[str, Any]) -> bool:
    return any(str(row.get(field) or "").strip() for field in ("title", "raw_text", "description"))


def _required_list(value: Any, default: list[str]) -> list[str]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_output_path(dataset_name: str) -> Path:
    return resolve_path(f"outputs/launch/paper_v1/dataset_readiness/{dataset_name}_readiness.json")
