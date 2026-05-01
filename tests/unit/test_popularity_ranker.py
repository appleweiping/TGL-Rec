from __future__ import annotations

from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.popularity import PopularityRanker


def test_popularity_ranker_uses_train_counts_and_item_tiebreak() -> None:
    train = [
        {"user_id": "u1", "item_id": "i2"},
        {"user_id": "u2", "item_id": "i2"},
        {"user_id": "u3", "item_id": "i1"},
    ]
    items = [{"item_id": "i1"}, {"item_id": "i2"}, {"item_id": "i3"}]
    ranker = PopularityRanker()
    ranker.fit(train, items)
    result = ranker.rank(
        RankingExample(
            user_id="u4",
            history=[],
            target_item="i3",
            candidate_items=["i3", "i1", "i2"],
        )
    )
    assert result.items == ["i2", "i1", "i3"]
