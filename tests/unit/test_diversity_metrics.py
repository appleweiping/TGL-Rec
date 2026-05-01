from llm4rec.metrics.diversity import aggregate_intra_list_diversity, catalog_coverage, item_coverage


def test_diversity_and_coverage_metrics():
    rows = [{"predicted_items": ["i1", "i2"]}, {"predicted_items": ["i2", "i3"]}]
    assert item_coverage(rows) == 3
    assert catalog_coverage(rows, {"i1", "i2", "i3", "i4"}) == 0.75
    features = {"i1": {"a"}, "i2": {"b"}, "i3": {"a"}}
    assert aggregate_intra_list_diversity(rows, item_features=features) > 0
