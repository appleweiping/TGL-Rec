from __future__ import annotations

from llm4rec.data.time_features import (
    DAY_SECONDS,
    HOUR_SECONDS,
    MONTH_SECONDS,
    WEEK_SECONDS,
    bucket_time_gap,
    consecutive_time_gaps,
    recency_blocks,
)


def test_time_gap_buckets() -> None:
    assert bucket_time_gap(0) == "same_session"
    assert bucket_time_gap(HOUR_SECONDS) == "same_session"
    assert bucket_time_gap(DAY_SECONDS) == "same_day"
    assert bucket_time_gap(WEEK_SECONDS) == "same_week"
    assert bucket_time_gap(MONTH_SECONDS) == "same_month"
    assert bucket_time_gap(MONTH_SECONDS + 1) == "old"


def test_consecutive_time_gaps_are_sorted() -> None:
    rows = [
        {"user_id": "u1", "item_id": "i2", "timestamp": 3},
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
    ]
    features = consecutive_time_gaps(rows)
    assert [row["item_id"] for row in features] == ["i1", "i2"]
    assert features[0]["gap_seconds"] is None
    assert features[1]["gap_seconds"] == 2.0


def test_recency_blocks() -> None:
    blocks = recency_blocks(["i1", "i2", "i3", "i4", "i5", "i6"])
    assert blocks["long_term"] == ["i1", "i2"]
    assert blocks["mid_term"] == ["i3", "i4"]
    assert blocks["short_term"] == ["i5", "i6"]
