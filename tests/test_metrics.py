import math

import pytest

from tglrec.eval.metrics import evaluate_rankings, hit_rate_at_k, mrr_at_k, ndcg_at_k, rank_by_score


def test_single_item_metrics():
    ranking = [10, 20, 30, 40]

    assert hit_rate_at_k(ranking, 30, 2) == 0.0
    assert hit_rate_at_k(ranking, 30, 3) == 1.0
    assert mrr_at_k(ranking, 30, 4) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranking, 30, 4) == pytest.approx(1 / math.log2(4))


def test_evaluate_rankings_aggregates_users():
    metrics = evaluate_rankings(
        rankings={"u1": ["a", "b"], "u2": ["b", "a"]},
        positives={"u1": "a", "u2": "a"},
        ks=(1, 2),
    )

    assert metrics["HR@1"] == pytest.approx(0.5)
    assert metrics["HR@2"] == pytest.approx(1.0)
    assert metrics["MRR@2"] == pytest.approx((1.0 + 0.5) / 2)


def test_evaluate_rankings_requires_all_users():
    with pytest.raises(ValueError, match="Missing rankings"):
        evaluate_rankings(rankings={"u1": ["a"]}, positives={"u1": "a", "u2": "b"})


def test_rank_by_score_breaks_integer_ties_by_numeric_item_id():
    assert rank_by_score({20: 0.8, 10: 0.8, 2: 0.8, 30: 0.7}) == [2, 10, 20, 30]
