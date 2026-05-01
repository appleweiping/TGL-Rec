from llm4rec.evaluation.segment import evaluate_segments, history_length_bucket


def test_segment_evaluation_by_domain():
    rows = [
        {"domain": "a", "predicted_items": ["i1"], "target_item": "i1"},
        {"domain": "b", "predicted_items": ["i2"], "target_item": "i3"},
    ]
    result = evaluate_segments(rows, ks=(1,), segment_fn=lambda row: row["domain"])
    assert {row["segment"] for row in result} == {"a", "b"}
    assert history_length_bucket(0) == "empty"
