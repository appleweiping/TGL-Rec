from __future__ import annotations

import pytest

from llm4rec.metrics.validity import aggregate_validity_metrics


def test_validity_counts_catalog_and_candidate_violations() -> None:
    rows = [
        {
            "candidate_items": ["i1", "i2"],
            "predicted_items": ["i1", "i3", "missing"],
        }
    ]
    metrics = aggregate_validity_metrics(
        rows,
        item_catalog={"i1", "i2", "i3"},
        candidate_protocol="full_catalog",
    )
    assert metrics["validity_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["hallucination_rate"] == pytest.approx(2.0 / 3.0)


def test_empty_predictions_are_not_hallucinations() -> None:
    metrics = aggregate_validity_metrics(
        [{"candidate_items": ["i1"], "predicted_items": []}],
        item_catalog={"i1"},
        candidate_protocol="full_catalog",
    )
    assert metrics == {"validity_rate": 1.0, "hallucination_rate": 0.0}
