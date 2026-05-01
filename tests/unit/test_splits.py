from __future__ import annotations

import pytest

from llm4rec.data.splits import build_user_histories, leave_one_out_split


def test_leave_one_out_is_deterministic() -> None:
    interactions = [
        {"user_id": "u1", "item_id": "i2", "timestamp": 2},
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i3", "timestamp": 3},
        {"user_id": "u2", "item_id": "i4", "timestamp": 1},
        {"user_id": "u2", "item_id": "i5", "timestamp": 2},
        {"user_id": "u2", "item_id": "i6", "timestamp": 3},
    ]
    labeled = leave_one_out_split(interactions)
    by_user = {(row["user_id"], row["item_id"]): row["split"] for row in labeled}
    assert by_user[("u1", "i1")] == "train"
    assert by_user[("u1", "i2")] == "valid"
    assert by_user[("u1", "i3")] == "test"
    assert build_user_histories(labeled)["u1"] == ["i1", "i2", "i3"]


def test_leave_one_out_requires_three_events() -> None:
    with pytest.raises(ValueError):
        leave_one_out_split(
            [
                {"user_id": "u1", "item_id": "i1", "timestamp": 1},
                {"user_id": "u1", "item_id": "i2", "timestamp": 2},
            ]
        )
