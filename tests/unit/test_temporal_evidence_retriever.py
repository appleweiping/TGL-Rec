from llm4rec.evidence.retriever import TemporalEvidenceRetriever


def test_retriever_returns_candidate_grounded_evidence():
    transition_edges = [
        {
            "source_item": "i1",
            "target_item": "i2",
            "count": 3,
            "user_count": 2,
            "mean_time_gap": 3600,
            "median_time_gap": 1800,
            "bucket_counts": {"same_session": 3},
        }
    ]
    time_window_edges = [
        {
            "source_item": "i1",
            "target_item": "i2",
            "count": 2,
            "user_count": 2,
            "weight": 0.8,
            "time_decayed_weight": 0.8,
            "mean_time_gap": 100,
            "median_time_gap": 100,
            "bucket_counts": {"same_session": 2},
            "directed": True,
            "window_seconds": 86400,
        }
    ]
    items = [
        {"item_id": "i1", "title": "Alpha", "category": "cat_a", "raw_text": "alpha cat a"},
        {"item_id": "i2", "title": "Beta", "category": "cat_b", "raw_text": "beta cat b"},
    ]
    retriever = TemporalEvidenceRetriever(
        transition_edges=transition_edges,
        time_window_edges=time_window_edges,
        item_records=items,
        config={
            "modes": ["transition_topk", "time_window_topk", "recent_history_focused"],
            "top_k_per_candidate": 5,
        },
        candidate_protocol="full_catalog",
    )
    result = retriever.retrieve(user_id="u1", history=["i1"], candidate_items=["i2"])
    assert {row.evidence_type for row in result.evidence} >= {"transition", "time_window", "history"}
    assert all(row.target_item == "i2" for row in result.evidence)
    assert result.metadata["constructed_from"] == "train_only"
