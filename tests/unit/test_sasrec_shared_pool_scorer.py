from __future__ import annotations

from llm4rec.models.sasrec import SASRecModel
from llm4rec.rankers.sasrec_shared_pool import SASRecSharedPoolScorer
from llm4rec.scoring.candidate_batch import CandidateBatch


def test_sasrec_shared_pool_scores_candidate_batch() -> None:
    torch = __import__("pytest").importorskip("torch")
    torch.manual_seed(0)
    model = SASRecModel(num_items=3, hidden_dim=4, num_layers=1, num_heads=1, dropout=0.0, max_seq_len=3)
    scorer = SASRecSharedPoolScorer(
        model=model.eval(),
        item_to_idx={"i1": 1, "i2": 2, "i3": 3},
        max_seq_len=3,
        torch_module=torch,
    )
    batch = CandidateBatch(
        user_ids=["u1", "u2"],
        histories=[["i1"], ["i2", "i1"]],
        target_items=["i2", "i3"],
        candidate_item_ids=[["i1", "i2"], ["i2", "i3"]],
        domains=[None, None],
        candidate_refs=[{}, {}],
        candidate_rows=[{}, {}],
        prediction_timestamps=[None, None],
    )

    scores = scorer.score_batch(batch)

    assert scores.shape == (2, 2)
