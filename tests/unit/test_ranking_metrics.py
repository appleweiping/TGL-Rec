from __future__ import annotations

import math

from llm4rec.metrics.ranking import (
    aggregate_ranking_metrics,
    coverage,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_perfect_ranking_metrics() -> None:
    predicted = ["i3", "i2", "i1"]
    assert recall_at_k(predicted, "i3", 1) == 1.0
    assert hit_rate_at_k(predicted, "i3", 1) == 1.0
    assert ndcg_at_k(predicted, "i3", 1) == 1.0
    assert mrr_at_k(predicted, "i3", 3) == 1.0


def test_target_missed_empty_and_duplicates() -> None:
    assert recall_at_k(["i1", "i1", "i2"], "i3", 3) == 0.0
    assert ndcg_at_k([], "i3", 5) == 0.0
    assert mrr_at_k(["i1", "i1", "i3"], "i3", 3) == 0.5


def test_aggregate_and_coverage() -> None:
    rows = [
        {"target_item": "i1", "predicted_items": ["i1", "i2"]},
        {"target_item": "i3", "predicted_items": ["i2", "i3"]},
    ]
    metrics = aggregate_ranking_metrics(rows, ks=(1, 2))
    assert metrics["Recall@1"] == 0.5
    assert metrics["HitRate@2"] == 1.0
    assert math.isclose(metrics["MRR@2"], 0.75)
    assert coverage(rows, {"i1", "i2", "i3", "i4"}) == 0.75
