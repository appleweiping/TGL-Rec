from __future__ import annotations

from llm4rec.rankers.mf_shared_pool import MFSharedPoolScorer
from llm4rec.scoring.candidate_batch import CandidateBatch


def test_mf_shared_pool_scores_batched_candidates() -> None:
    torch = __import__("pytest").importorskip("torch")
    user_emb = torch.nn.Embedding(2, 2)
    item_emb = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        user_emb.weight[:] = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        item_emb.weight[:] = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.9, 0.0], [0.2, 0.0]])
    scorer = MFSharedPoolScorer(
        user_emb=user_emb,
        item_emb=item_emb,
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2, "i3": 3},
        torch_module=torch,
    )
    batch = CandidateBatch(
        user_ids=["u1"],
        histories=[[]],
        target_items=["i2"],
        candidate_item_ids=[["i1", "i2", "i3"]],
        domains=[None],
        candidate_refs=[{}],
        candidate_rows=[{}],
        prediction_timestamps=[None],
    )

    scores = scorer.score_batch(batch)

    assert scores.shape == (1, 3)
    assert float(scores[0, 1]) > float(scores[0, 2]) > float(scores[0, 0])
