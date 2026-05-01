"""Simple file logger for reproducible run directories."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    """Append-only text logger for a single run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8", newline="\n")

    def info(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{timestamp} INFO {message}\n")
