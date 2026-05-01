"""JSON checkpoint helpers for small deterministic baseline smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json


class CheckpointError(RuntimeError):
    """Raised when checkpoint save/load semantics are invalid."""


def save_checkpoint(path: str | Path, payload: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Save a JSON checkpoint with explicit overwrite semantics."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise CheckpointError(f"Checkpoint already exists and overwrite=false: {output}")
    write_json(output, payload)
    return output


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a JSON checkpoint payload."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointError(f"Missing checkpoint: {checkpoint}")
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CheckpointError(f"Checkpoint payload must be an object: {checkpoint}")
    return data
