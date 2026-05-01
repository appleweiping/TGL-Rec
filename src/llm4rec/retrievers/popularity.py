"""Popularity candidate retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.retrievers.base import RetrievalResult


class PopularityRetriever:
    """Retrieve globally popular items from train interactions."""

    name = "popularity"

    def __init__(self) -> None:
        self.ranker = PopularityRanker()
        self.item_ids: list[str] = []

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        self.item_ids = sorted(str(row["item_id"]) for row in item_records)
        self.ranker.fit(train_interactions, item_records)

    def retrieve(
        self,
        *,
        user_id: str,
        history: list[str],
        top_k: int,
        domain: str | None = None,
    ) -> RetrievalResult:
        example = RankingExample(
            user_id=user_id,
            history=history,
            target_item="",
            candidate_items=self.item_ids,
            domain=domain,
        )
        result = self.ranker.rank(example)
        return RetrievalResult(
            user_id=user_id,
            items=result.items[:top_k],
            scores=result.scores[:top_k],
            metadata={"retriever": self.name},
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        self.ranker.save_artifact(output_dir)
