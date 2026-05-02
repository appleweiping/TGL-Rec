"""Artifact I/O helpers for reproducible smoke runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:  # Prefer PyYAML when installed.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised by bare Python smoke commands.
    yaml = None


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object.")
            rows.append(value)
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream a JSONL file as dictionaries."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object.")
            yield value


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL with deterministic key ordering."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write YAML with stable key ordering disabled for readability."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    else:
        text = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8", newline="\n")


def write_metric_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write metric rows in long format."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "domain", "metric", "value"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    """Write arbitrary dictionaries to CSV with deterministic columns."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames
    if columns is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row)
        columns = sorted(keys)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in columns})
