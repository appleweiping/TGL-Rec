"""Environment capture for run artifacts."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def current_git_commit(repo_dir: str | Path = ".") -> str | None:
    """Return the current git commit, or None when git is unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_environment(repo_dir: str | Path = ".") -> dict[str, Any]:
    """Collect minimal environment metadata for reproducibility."""

    return {
        "git_commit": current_git_commit(repo_dir),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
    }
