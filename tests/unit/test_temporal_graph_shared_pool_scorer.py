from __future__ import annotations

from llm4rec.encoders.temporal_graph_encoder import TemporalGraphEncoder
from llm4rec.rankers.temporal_graph_shared_pool import TemporalGraphSharedPoolScorer
from llm4rec.scoring.candidate_batch import CandidateBatch


def test_temporal_graph_shared_pool_scores_batch() -> None:
    torch = __import__("pytest").importorskip("torch")
    encoder = TemporalGraphEncoder(
        num_users=1,
        num_items=2,
        hidden_dim=4,
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2},
    ).eval()
    scorer = TemporalGraphSharedPoolScorer(encoder=encoder, torch_module=torch)
    batch = CandidateBatch(
        user_ids=["u1"],
        histories=[[]],
        target_items=["i2"],
        candidate_item_ids=[["i1", "i2"]],
        domains=[None],
        candidate_refs=[{}],
        candidate_rows=[{}],
        prediction_timestamps=[123.0],
    )

    scores = scorer.score_batch(batch)

    assert scores.shape == (1, 2)
