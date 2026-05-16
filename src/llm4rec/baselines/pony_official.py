"""Pony official same-candidate baseline manifest helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_MANIFEST_PATH = Path("configs/baselines/pony_official_external.yaml")
REQUIRED_SCORE_SCHEMA = ["source_event_id", "user_id", "item_id", "score"]


def load_pony_official_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load the active Pony official baseline manifest."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {manifest_path}")
    return data


def pony_official_baseline_names(path: str | Path = DEFAULT_MANIFEST_PATH) -> list[str]:
    """Return Pony official baseline IDs in manifest order."""

    manifest = load_pony_official_manifest(path)
    baselines = manifest.get("official_baselines", {})
    if not isinstance(baselines, dict):
        raise ValueError("official_baselines must be a mapping")
    return list(baselines)


def validate_pony_official_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Validate the lightweight manifest contract without touching large artifacts."""

    manifest = load_pony_official_manifest(path)
    errors: list[str] = []
    schema = manifest.get("score_file_contract", {}).get("required_schema")
    if schema != REQUIRED_SCORE_SCHEMA:
        errors.append(f"required_schema must be {REQUIRED_SCORE_SCHEMA}")
    baselines = manifest.get("official_baselines", {})
    if not isinstance(baselines, dict) or not baselines:
        errors.append("official_baselines must be a non-empty mapping")
        baselines = {}
    for name, spec in baselines.items():
        if not isinstance(spec, dict):
            errors.append(f"{name}: spec must be a mapping")
            continue
        for field in ("method_id", "status", "official_repo", "pinned_commit"):
            if not spec.get(field):
                errors.append(f"{name}: missing {field}")
        status = str(spec.get("status", ""))
        if status == "official_completed_declared_domains":
            domains = spec.get("domains")
            if domains != ["beauty", "books", "electronics", "movies"]:
                errors.append(f"{name}: completed baselines must list all four domains")
            archives = spec.get("evidence_archives")
            if not isinstance(archives, dict) or sorted(archives) != [
                "beauty",
                "books",
                "electronics",
                "movies",
            ]:
                errors.append(f"{name}: completed baselines must record four evidence_archives")
        if status == "official_pending_completion":
            if not spec.get("domains_pending"):
                errors.append(f"{name}: pending baselines must record domains_pending")
        if not (
            spec.get("summary_csv")
            or spec.get("evidence_archives")
            or spec.get("blocker")
        ):
            errors.append(f"{name}: must record summary_csv, evidence_archives, or blocker")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "baseline_count": len(baselines),
        "completed": [
            name
            for name, spec in baselines.items()
            if spec.get("status") == "official_completed_declared_domains"
        ],
        "pending": [
            name
            for name, spec in baselines.items()
            if spec.get("status") == "official_pending_completion"
        ],
        "blocked": [
            name
            for name, spec in baselines.items()
            if spec.get("status") == "official_blocked_replaced"
        ],
        "status": "pass",
    }
