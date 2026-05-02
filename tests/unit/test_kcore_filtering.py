from llm4rec.data.filtering import filter_by_min_counts
from llm4rec.data.kcore import iterative_k_core


def test_iterative_k_core_removes_cascading_low_count_rows():
    rows = [
        {"domain": "D", "item_id": "I1", "timestamp": 1, "user_id": "U1"},
        {"domain": "D", "item_id": "I2", "timestamp": 2, "user_id": "U1"},
        {"domain": "D", "item_id": "I3", "timestamp": 3, "user_id": "U1"},
        {"domain": "D", "item_id": "I1", "timestamp": 4, "user_id": "U2"},
        {"domain": "D", "item_id": "I2", "timestamp": 5, "user_id": "U2"},
        {"domain": "D", "item_id": "I1", "timestamp": 6, "user_id": "U3"},
    ]

    filtered, report = iterative_k_core(rows, user_k=2, item_k=2)

    assert len(filtered) == 4
    assert {row["user_id"] for row in filtered} == {"U1", "U2"}
    assert {row["item_id"] for row in filtered} == {"I1", "I2"}
    assert report["users_still_below_threshold"] == 0
    assert report["items_still_below_threshold"] == 0
    assert report["num_iterations"] == 2


def test_single_pass_filter_reports_remaining_threshold_violations():
    rows = [
        {"domain": "D", "item_id": "I1", "timestamp": 1, "user_id": "U1"},
        {"domain": "D", "item_id": "I2", "timestamp": 2, "user_id": "U1"},
        {"domain": "D", "item_id": "I1", "timestamp": 3, "user_id": "U2"},
    ]

    filtered, report = filter_by_min_counts(rows, user_min_interactions=2, item_min_interactions=2)

    assert filtered == [{"domain": "D", "item_id": "I1", "timestamp": 1, "user_id": "U1"}]
    assert report["users_still_below_threshold"] == 1
    assert report["items_still_below_threshold"] == 1
