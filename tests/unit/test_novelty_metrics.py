from llm4rec.metrics.novelty import aggregate_novelty, item_novelty


def test_novelty_higher_for_less_popular_item():
    popularity = {"head": 10, "tail": 1}
    assert item_novelty("tail", popularity) > item_novelty("head", popularity)
    assert aggregate_novelty([{"predicted_items": ["tail"]}], popularity) > 0
