"""Logging helpers for reproducible runs and preprocessing artifacts."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tglrec.utils.config import write_config
from tglrec.utils.io import ensure_dir, write_json


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise root logger for CLI use."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def current_git_commit(repo_dir: str | Path = ".") -> str:
    """Return the current git commit, or a clear sentinel outside git repos."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE: not a git repository"
    return result.stdout.strip()


def write_artifact_manifest(
    output_dir: str | Path,
    *,
    command: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
    repo_dir: str | Path = ".",
) -> None:
    """Write the common manifest files expected by the research workflow."""

    root = ensure_dir(output_dir)
    write_config(config, root / "config.yaml")
    write_json(metadata, root / "metadata.json")
    (root / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    (root / "git_commit.txt").write_text(
        current_git_commit(repo_dir) + "\n", encoding="utf-8", newline="\n"
    )
    (root / "created_at_utc.txt").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8", newline="\n"
    )
