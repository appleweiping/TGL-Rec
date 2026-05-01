"""Retriever contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievalResult:
    """Candidate retrieval output."""

    user_id: str
    items: list[str]
    scores: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(Protocol):
    """Minimal retriever interface."""

    name: str

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        """Fit retriever state from train-only evidence."""

    def retrieve(
        self,
        *,
        user_id: str,
        history: list[str],
        top_k: int,
        domain: str | None = None,
    ) -> RetrievalResult:
        """Return top-k candidate items."""

    def save_artifact(self, output_dir: str | Path) -> None:
        """Persist optional retriever artifacts."""
