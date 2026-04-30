"""Small IO helpers shared across data and experiment modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data

