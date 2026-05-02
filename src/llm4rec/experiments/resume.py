"""Resume and failure policy contracts for planned paper jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResumePolicy:
    """Structured resume policy saved in launch manifests."""

    allow_resume: bool = True
    resume_from_checkpoints: bool = True
    resume_from_predictions: bool = True
    overwrite_complete_runs: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_resume": self.allow_resume,
            "overwrite_complete_runs": self.overwrite_complete_runs,
            "resume_from_checkpoints": self.resume_from_checkpoints,
            "resume_from_predictions": self.resume_from_predictions,
        }
