"""Smoke-only dynamic graph encoder interface implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.encoders.base import BaseDynamicGraphEncoder


class TemporalMemoryEncoderStub(BaseDynamicGraphEncoder):
    """Deterministic non-reportable temporal memory stub for smoke tests only."""

    reportable = False

    def __init__(self, *, dimension: int = 4) -> None:
        self.dimension = int(dimension)
        self.user_counts: dict[str, int] = {}
        self.item_counts: dict[str, int] = {}

    def fit(
        self,
        events: list[dict[str, Any]],
        item_features: dict[str, Any] | None = None,
        user_features: dict[str, Any] | None = None,
    ) -> None:
        del item_features, user_features
        for event in events:
            self.update(event)

    def encode_user(self, user_id: str, timestamp: int | float | None) -> list[float]:
        return _vector(self.user_counts.get(str(user_id), 0), timestamp, self.dimension)

    def encode_item(self, item_id: str, timestamp: int | float | None) -> list[float]:
        return _vector(self.item_counts.get(str(item_id), 0), timestamp, self.dimension)

    def update(self, event: dict[str, Any]) -> None:
        user_id = str(event.get("user_id", ""))
        item_id = str(event.get("item_id", ""))
        if user_id:
            self.user_counts[user_id] = self.user_counts.get(user_id, 0) + 1
        if item_id:
            self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "item_counts": self.item_counts,
                    "reportable": self.reportable,
                    "user_counts": self.user_counts,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemporalMemoryEncoderStub":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        encoder = cls(dimension=int(data.get("dimension", 4)))
        encoder.user_counts = {str(key): int(value) for key, value in data.get("user_counts", {}).items()}
        encoder.item_counts = {str(key): int(value) for key, value in data.get("item_counts", {}).items()}
        return encoder


def _vector(count: int, timestamp: int | float | None, dimension: int) -> list[float]:
    ts = 0.0 if timestamp is None else float(timestamp)
    base = float(count)
    values = [base, ts]
    while len(values) < dimension:
        values.append((base + len(values)) / float(max(1, dimension)))
    return values[:dimension]
