"""Result-locking checks for completed paper runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.io.artifacts import write_json


class ResultLockError(ValueError):
    """Raised when an incomplete run is asked to lock."""


REQUIRED_RESULT_FILES = ["resolved_config.yaml", "predictions.jsonl", "metrics.json", "metrics.csv"]


def lock_results(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Create a lock manifest only for complete runs."""

    run = Path(run_dir)
    missing = [name for name in REQUIRED_RESULT_FILES if not (run / name).is_file()]
    if missing:
        raise ResultLockError(f"Refusing to lock incomplete run {run}: missing {missing}")
    manifest = {
        NO_EXECUTION_FLAG: False,
        "locked": True,
        "required_files": REQUIRED_RESULT_FILES,
        "run_dir": str(run.resolve()),
    }
    if output_path is not None:
        write_json(output_path, manifest)
    return manifest
