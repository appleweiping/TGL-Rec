from __future__ import annotations

from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.bm25 import BM25Ranker


def test_bm25_ranker_uses_history_text() -> None:
    items = [
        {
            "item_id": "i1",
            "title": "space opera",
            "description": "",
            "category": "sci_fi",
            "brand": None,
            "domain": "tiny",
            "raw_text": "space opera starship",
        },
        {
            "item_id": "i2",
            "title": "space sequel",
            "description": "",
            "category": "sci_fi",
            "brand": None,
            "domain": "tiny",
            "raw_text": "space starship sequel",
        },
        {
            "item_id": "i3",
            "title": "cooking guide",
            "description": "",
            "category": "food",
            "brand": None,
            "domain": "tiny",
            "raw_text": "recipe kitchen",
        },
    ]
    ranker = BM25Ranker()
    ranker.fit([], items)
    result = ranker.rank(
        RankingExample(
            user_id="u1",
            history=["i1"],
            target_item="i2",
            candidate_items=["i3", "i2"],
            domain="tiny",
        )
    )
    assert result.items[0] == "i2"
    assert result.scores[0] > result.scores[1]
