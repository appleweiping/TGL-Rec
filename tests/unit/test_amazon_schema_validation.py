from llm4rec.data.schema_validation import normalize_interaction, normalize_item


def test_normalize_interaction_accepts_amazon_aliases():
    row, reason = normalize_interaction(
        {"overall": "5", "parent_asin": "B1", "timestamp": 1700000000000, "user_id": "U1"},
        "Beauty",
    )
    assert reason is None
    assert row == {"domain": "Beauty", "item_id": "B1", "rating": 5.0, "timestamp": 1700000000, "user_id": "U1"}


def test_normalize_item_builds_raw_text_from_metadata():
    row, reason = normalize_item(
        {"categories": [["Beauty", "Skin"]], "features": ["soft"], "parent_asin": "B1", "store": "Store", "title": "Item"},
        "Beauty",
    )
    assert reason is None
    assert row["item_id"] == "B1"
    assert row["brand"] == "Store"
    assert "Item" in row["raw_text"]
    assert "Beauty" in row["raw_text"]
