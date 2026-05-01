"""Train-only popularity ranker."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json
from llm4rec.rankers.base import RankingExample, RankingResult, result_from_scores


class PopularityRanker:
    """Rank candidates by train-interaction count with item-id tie-breaks."""

    name = "popularity"

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        self.counts = Counter(str(row["item_id"]) for row in train_interactions)
        for item in item_records:
            self.counts.setdefault(str(item["item_id"]), 0)

    def rank(self, example: RankingExample) -> RankingResult:
        scores = {str(item_id): float(self.counts.get(str(item_id), 0)) for item_id in example.candidate_items}
        return result_from_scores(
            example=example,
            scores_by_item=scores,
            metadata={"fit_on": "train_interactions", "tie_break": "item_id"},
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        write_json(Path(output_dir) / "popularity_counts.json", dict(sorted(self.counts.items())))
