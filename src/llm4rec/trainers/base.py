"""Training contracts for pre-experiment baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TrainingResult:
    """Result of a local training or dry-run planning command."""

    run_dir: Path
    checkpoint_path: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(Protocol):
    """Minimal trainer protocol."""

    def train(self) -> TrainingResult:
        """Run training or dry-run planning."""
