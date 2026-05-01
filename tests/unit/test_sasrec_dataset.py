from llm4rec.trainers.sasrec import SASRecSequenceDataset, build_item_mappings, sample_negative_items


def test_sasrec_negative_sampling_is_deterministic():
    kwargs = {
        "all_items": ["i1", "i2", "i3", "i4"],
        "positives": {"i1", "i2"},
        "num_negatives": 3,
        "seed": 2026,
    }
    assert sample_negative_items(**kwargs) == sample_negative_items(**kwargs)
    assert set(sample_negative_items(**kwargs)).isdisjoint({"i1", "i2"})


def test_sasrec_dataset_uses_train_sequences_only():
    items = [{"item_id": f"i{index}"} for index in range(1, 6)]
    item_to_idx, _idx_to_item = build_item_mappings(items)
    interactions = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2},
        {"user_id": "u1", "item_id": "i3", "timestamp": 3},
    ]
    dataset = SASRecSequenceDataset(
        train_interactions=interactions,
        item_to_idx=item_to_idx,
        max_seq_len=4,
        num_negatives=1,
        seed=1,
    )
    assert len(dataset) == 2
    assert dataset[0].positive_index == item_to_idx["i2"]
    assert dataset[1].positive_index == item_to_idx["i3"]
