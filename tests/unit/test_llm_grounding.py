from llm4rec.diagnostics.llm_grounding import build_edge_index, evaluate_evidence_grounding


def test_grounding_checks_transition_and_item_ids():
    transition_edges = build_edge_index([{"source_item": "i1", "target_item": "i2"}])
    result = evaluate_evidence_grounding(
        [
            {
                "type": "transition",
                "source_item": "i1",
                "target_item": "i2",
                "text": "i1 to i2",
            },
            {
                "type": "transition",
                "source_item": "i9",
                "target_item": "i2",
                "text": "bad",
            },
        ],
        history_items=["i1"],
        candidate_items=["i2"],
        transition_edges=transition_edges,
        time_window_edges={},
    )

    assert result["evidence_count"] == 2
    assert result["grounded_evidence_count"] == 1
    assert result["transition_evidence_usage"]

