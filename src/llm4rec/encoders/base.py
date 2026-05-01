"""Encoder interfaces for future graph-aware recommendation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDynamicGraphEncoder(ABC):
    """Interface for future TGN/DGSR-style dynamic graph encoders."""

    reportable: bool = False

    @abstractmethod
    def fit(
        self,
        events: list[dict[str, Any]],
        item_features: dict[str, Any] | None = None,
        user_features: dict[str, Any] | None = None,
    ) -> None:
        """Fit encoder state from train-only events."""

    @abstractmethod
    def encode_user(self, user_id: str, timestamp: int | float | None) -> list[float]:
        """Encode one user at a timestamp."""

    @abstractmethod
    def encode_item(self, item_id: str, timestamp: int | float | None) -> list[float]:
        """Encode one item at a timestamp."""

    @abstractmethod
    def update(self, event: dict[str, Any]) -> None:
        """Update memory with one event."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist encoder state."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseDynamicGraphEncoder":
        """Load encoder state."""
