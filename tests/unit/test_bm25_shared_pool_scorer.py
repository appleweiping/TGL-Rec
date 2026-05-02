from __future__ import annotations

from llm4rec.scoring.candidate_batch import CandidateBatch
from llm4rec.rankers.bm25_shared_pool import BM25SharedPoolScorer


def test_bm25_shared_pool_scores_candidate_matrix() -> None:
    scorer = BM25SharedPoolScorer()
    scorer.fit(
        train_interactions=[{"user_id": "u1", "item_id": "i1"}],
        item_records=[
            {"item_id": "i1", "title": "space laser", "description": None},
            {"item_id": "i2", "title": "space ship", "description": None},
            {"item_id": "i3", "title": "garden hose", "description": None},
        ],
    )
    batch = CandidateBatch(
        user_ids=["u1"],
        histories=[["i1"]],
        target_items=["i2"],
        candidate_item_ids=[["i2", "i3"]],
        domains=[None],
        candidate_refs=[{}],
        candidate_rows=[{}],
        prediction_timestamps=[None],
    )

    scores = scorer.score_batch(batch)

    assert scores[0][0] > scores[0][1]
    assert scorer.metadata["vocabulary_size"] > 0
