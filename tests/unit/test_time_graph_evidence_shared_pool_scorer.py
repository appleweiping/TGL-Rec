from __future__ import annotations

from collections import Counter

from llm4rec.rankers.time_graph_evidence_shared_pool import TimeGraphEvidenceSharedPoolScorer
from llm4rec.scoring.candidate_batch import CandidateBatch


def test_time_graph_evidence_shared_pool_scores_sparse_evidence_only() -> None:
    scorer = TimeGraphEvidenceSharedPoolScorer(
        transition_counts=Counter({("i1", "i3"): 2}),
        category_by_item={"i1": "space", "i2": "garden", "i3": "space"},
        candidate_pool_items=["i2", "i3"],
    )
    batch = CandidateBatch(
        user_ids=["u1"],
        histories=[["i1"]],
        target_items=["i3"],
        candidate_item_ids=[["i2", "i3"]],
        domains=[None],
        candidate_refs=[{}],
        candidate_rows=[{}],
        prediction_timestamps=[None],
    )

    scores = scorer.score_batch(batch)
    metadata = scorer.metadata_for_top_items(top_items=["i3"], top_scores=[float(scores[0, 1])], limit=1)

    assert float(scores[0, 1]) > float(scores[0, 0])
    assert list(metadata["top_evidence"]) == ["i3"]
