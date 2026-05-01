from llm4rec.metrics.long_tail import long_tail_items, long_tail_ratio, popularity_bucket_metrics, popularity_counts


def test_long_tail_ratio_and_buckets():
    counts = popularity_counts([{"item_id": "i1"}, {"item_id": "i1"}, {"item_id": "i2"}])
    tail = long_tail_items(counts, quantile=0.5)
    rows = [{"predicted_items": ["i2", "i1"]}]
    assert long_tail_ratio(rows, tail) == 0.5
    buckets = popularity_bucket_metrics(rows, counts)
    assert buckets["popularity_bucket_tail_rate"] > 0
