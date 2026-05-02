"""Amazon Reviews 2023 raw-domain discovery and schema inspection."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.data.schema_validation import (
    can_convert_item_fields,
    can_convert_review_fields,
    detected_fields,
)
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json

SUPPORTED_SUFFIXES = [".jsonl", ".jsonl.gz", ".parquet", ".csv"]


def inspect_amazon_reviews_2023(
    config_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rows: int = 1000,
) -> dict[str, Any]:
    """Inspect raw domain directories and write a schema report."""

    config = load_yaml_config(config_path)
    raw_domains = _raw_domains_from_config(config)
    domain_reports = {}
    for domain, raw_path in raw_domains.items():
        domain_reports[domain] = inspect_domain(domain, resolve_path(raw_path), sample_rows=sample_rows)
    report = {
        NO_EXECUTION_FLAG: True,
        "config_path": str(resolve_path(config_path)),
        "domains": domain_reports,
        "overall_status": "convertible" if all(row["can_convert"] for row in domain_reports.values()) else "partial",
    }
    output = _schema_output_path(config, config_path, output_path)
    write_json(output, report)
    return report


def inspect_domain(domain: str, path: str | Path, *, sample_rows: int = 1000) -> dict[str, Any]:
    """Inspect one Amazon Reviews 2023 domain directory."""

    directory = Path(path)
    if not directory.is_dir():
        return {
            "can_convert": False,
            "detected_fields": {"items": [], "reviews": []},
            "files_found": [],
            "metadata_file_candidate": None,
            "path": str(directory),
            "review_file_candidate": None,
            "status": "MISSING",
            "warnings": [f"missing directory: {directory}"],
        }
    files = sorted([file for file in directory.iterdir() if file.is_file()], key=lambda value: value.name)
    review_file = choose_review_file(files)
    metadata_file = choose_metadata_file(files)
    warnings: list[str] = []
    review_sample = _sample_rows(review_file, sample_rows) if review_file else []
    item_sample = _sample_rows(metadata_file, sample_rows) if metadata_file else []
    review_fields = detected_fields(review_sample)
    item_fields = detected_fields(item_sample)
    if review_file is None:
        warnings.append("no review/interactions file candidate found")
    if metadata_file is None:
        warnings.append("no metadata/items file candidate found")
    can_convert = bool(review_file) and can_convert_review_fields(review_fields) and bool(metadata_file) and can_convert_item_fields(item_fields)
    return {
        "can_convert": can_convert,
        "compression_format": {
            "items": _format(metadata_file) if metadata_file else None,
            "reviews": _format(review_file) if review_file else None,
        },
        "detected_fields": {
            "items": item_fields,
            "reviews": review_fields,
        },
        "files_found": [
            {"format": _format(file), "name": file.name, "size_bytes": file.stat().st_size}
            for file in files
        ],
        "metadata_file_candidate": str(metadata_file) if metadata_file else None,
        "path": str(directory),
        "review_file_candidate": str(review_file) if review_file else None,
        "row_count_estimate": {
            "items": _estimate_row_count(metadata_file, sample_rows) if metadata_file else None,
            "reviews": _estimate_row_count(review_file, sample_rows) if review_file else None,
        },
        "status": "READY_TO_CONVERT" if can_convert else "PARTIAL",
        "warnings": warnings,
    }


def choose_review_file(files: list[Path]) -> Path | None:
    candidates = [file for file in files if _format(file) in {"jsonl", "jsonl.gz", "parquet", "csv"} and not file.name.startswith("meta_")]
    return _prefer_uncompressed(candidates)


def choose_metadata_file(files: list[Path]) -> Path | None:
    candidates = [file for file in files if _format(file) in {"jsonl", "jsonl.gz", "parquet", "csv"} and file.name.startswith("meta_")]
    return _prefer_uncompressed(candidates)


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream records from supported file formats."""

    file = Path(path)
    fmt = _format(file)
    if fmt == "jsonl":
        with file.open("r", encoding="utf-8") as handle:
            yield from _iter_json_lines(handle)
    elif fmt == "jsonl.gz":
        with gzip.open(file, "rt", encoding="utf-8") as handle:
            yield from _iter_json_lines(handle)
    elif fmt == "csv":
        with file.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield dict(row)
    elif fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional env.
            raise RuntimeError("Parquet Amazon files require pyarrow. Install pyarrow or provide JSONL/CSV files.") from exc
        table = pq.read_table(file)
        for row in table.to_pylist():
            yield dict(row)
    else:
        raise ValueError(f"Unsupported Amazon Reviews 2023 file format: {file}")


def _sample_rows(path: Path | None, limit: int) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = []
    try:
        for row in iter_records(path):
            rows.append(row)
            if len(rows) >= limit:
                break
    except Exception:
        return []
    return rows


def _estimate_row_count(path: Path | None, sample_rows: int) -> int | None:
    if path is None:
        return None
    fmt = _format(path)
    if fmt in {"csv", "parquet"}:
        return None
    count = 0
    total_bytes = 0
    opener = gzip.open if fmt == "jsonl.gz" else open
    mode = "rb"
    with opener(path, mode) as handle:  # type: ignore[arg-type]
        for line in handle:
            count += 1
            total_bytes += len(line)
            if count >= sample_rows:
                break
    if count == 0 or total_bytes == 0:
        return 0
    if fmt == "jsonl.gz":
        return None
    return int(path.stat().st_size / (total_bytes / count))


def _iter_json_lines(handle: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if isinstance(value, dict):
            yield value


def _format(path: Path | None) -> str:
    if path is None:
        return "missing"
    name = path.name.lower()
    if name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".csv"):
        return "csv"
    return "other"


def _prefer_uncompressed(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    rank = {"jsonl": 0, "csv": 1, "parquet": 2, "jsonl.gz": 3, "other": 4}
    return sorted(candidates, key=lambda file: (rank.get(_format(file), 9), file.name))[0]


def _raw_domains_from_config(config: dict[str, Any]) -> dict[str, str]:
    dataset = dict(config.get("dataset", config))
    raw_domains = dataset.get("raw_domains", config.get("raw_domains", {}))
    if (not raw_domains) and dataset.get("source_config"):
        source = load_yaml_config(dataset["source_config"])
        source_dataset = dict(source.get("dataset", source))
        raw_domains = source_dataset.get("raw_domains", source.get("raw_domains", {}))
    if not isinstance(raw_domains, dict) or not raw_domains:
        raise ValueError("Amazon Reviews 2023 config must declare raw_domains")
    return {str(domain): str(path) for domain, path in raw_domains.items()}


def _schema_output_path(config: dict[str, Any], config_path: str | Path, output_path: str | Path | None) -> Path:
    if output_path is not None:
        return resolve_path(output_path)
    dataset = dict(config.get("dataset", config))
    paths = dict(dataset.get("paths", {}))
    if paths.get("schema_report"):
        return resolve_path(paths["schema_report"])
    readiness = dict(config.get("readiness", {}))
    if readiness.get("schema_report_path"):
        return resolve_path(readiness["schema_report_path"])
    return resolve_path(Path(config_path).with_name("amazon_reviews_2023_schema_report.json"))


def save_inspection_summary(report: dict[str, Any], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    write_json(output_path, report)
