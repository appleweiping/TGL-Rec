from __future__ import annotations

from llm4rec.scoring.topk import top_k_items_and_scores
from llm4rec.scoring.vectorized import adjusted_batch_size, estimate_score_tensor_memory_mb


def test_topk_breaks_ties_by_item_id_ascending() -> None:
    items, scores = top_k_items_and_scores([0.5, 0.7, 0.7, 0.1], ["b", "z", "a", "c"], top_n=3)

    assert items == ["a", "z", "b"]
    assert scores == [0.7, 0.7, 0.5]


def test_memory_guard_reduces_batch_size() -> None:
    safe = adjusted_batch_size(
        requested_batch_size=512,
        candidate_size=1000,
        score_dtype="float32",
        max_memory_mb=1.0,
    )

    assert safe < 512
    assert estimate_score_tensor_memory_mb(
        batch_size=safe,
        candidate_size=1000,
        score_dtype="float32",
    ) <= 1.0
