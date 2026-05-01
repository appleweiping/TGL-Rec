"""Deterministic random sanity ranker."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from llm4rec.rankers.base import RankingExample, RankingResult


class RandomRanker:
    """Deterministic random ranker for sanity checks, not a strong baseline."""

    name = "random"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        self.item_count = len(item_records)

    def rank(self, example: RankingExample) -> RankingResult:
        scored = {
            str(item_id): self._score(example.user_id, str(item_id))
            for item_id in example.candidate_items
        }
        ordered = sorted(scored, key=lambda item_id: (-scored[item_id], item_id))
        return RankingResult(
            user_id=example.user_id,
            items=ordered,
            scores=[scored[item_id] for item_id in ordered],
            metadata={"non_reportable": True, "seed": self.seed},
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        return None

    def _score(self, user_id: str, item_id: str) -> float:
        key = f"{self.seed}|{user_id}|{item_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return int(digest[:16], 16) / float(16**16)
