from __future__ import annotations

from llm4rec.diagnostics.statistics import (
    compare_prediction_sets,
    compute_metric_deltas,
    prediction_overlap_at_k,
    target_rank,
)


def test_prediction_overlap_and_rank() -> None:
    assert target_rank(["i2", "i1", "i1"], "i1") == 2
    assert prediction_overlap_at_k(["i1", "i2"], ["i2", "i3"], k=2) == 1 / 3


def test_metric_delta_computation() -> None:
    rows = [
        {"method": "bm25", "perturbation": "original", "Recall@5": 0.5, "NDCG@5": 0.25},
        {"method": "bm25", "perturbation": "reversed", "Recall@5": 0.25, "NDCG@5": 0.1},
    ]
    deltas = compute_metric_deltas(rows)
    reversed_row = next(row for row in deltas if row["perturbation"] == "reversed")
    assert reversed_row["delta_Recall@5_vs_original"] == -0.25
    assert reversed_row["delta_NDCG@5_vs_original"] == -0.15


def test_compare_prediction_sets() -> None:
    original = [
        {
            "method": "m",
            "predicted_items": ["i1", "i2", "i3"],
            "target_item": "i3",
            "user_id": "u1",
        }
    ]
    variant = [
        {
            "method": "m",
            "predicted_items": ["i3", "i2", "i1"],
            "target_item": "i3",
            "user_id": "u1",
        }
    ]
    stats = compare_prediction_sets(original, variant, k=2)
    assert stats["mean_target_rank_shift"] == -2.0
    assert stats["mean_prediction_overlap"] == 1 / 3
