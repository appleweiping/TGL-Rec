"""Registry for planned paper-scale split/candidate artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.io.artifacts import write_json


@dataclass
class ArtifactRegistry:
    """Small deterministic registry for planned artifacts."""

    protocol_version: str
    entries: list[dict[str, Any]] = field(default_factory=list)

    def register(self, *, dataset: str, artifact_type: str, path: str | Path, status: str = "planned") -> None:
        self.entries.append(
            {
                "artifact_type": artifact_type,
                "dataset": dataset,
                "path": str(path),
                "protocol_version": self.protocol_version,
                "status": status,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            NO_EXECUTION_FLAG: True,
            "artifacts": sorted(self.entries, key=lambda row: (row["dataset"], row["artifact_type"], row["path"])),
            "protocol_version": self.protocol_version,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())
