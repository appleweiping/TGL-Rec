"""BM25 candidate retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.retrievers.base import RetrievalResult


class BM25Retriever:
    """Retrieve candidates by local BM25 over item text."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.ranker = BM25Ranker(k1=k1, b=b)
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
