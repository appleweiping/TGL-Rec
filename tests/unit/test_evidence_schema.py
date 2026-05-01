import json

import pytest

from llm4rec.evidence.base import Evidence, EvidenceSchemaError


def test_evidence_schema_is_json_serializable():
    evidence = Evidence(
        evidence_id="ev_test",
        evidence_type="transition",
        source_item="i1",
        target_item="i2",
        support_items=["i1", "i2"],
        timestamp_info={
            "source_timestamp": None,
            "target_timestamp": None,
            "mean_gap_seconds": 3600,
            "median_gap_seconds": 1800,
            "gap_bucket": "same_session",
        },
        stats={
            "transition_count": 5,
            "user_count": 4,
            "time_window_score": 0.7,
            "semantic_similarity": 0.2,
            "time_decayed_weight": 0.5,
        },
        text="Users who watched i1 often watched i2 within the same session.",
        provenance={
            "graph_artifact": "transition_edges.jsonl",
            "split": "train",
            "candidate_protocol": "fixed",
            "constructed_from": "train_only",
        },
        metadata={},
    )
    payload = evidence.to_dict()
    assert json.loads(json.dumps(payload))["evidence_id"] == "ev_test"
    assert Evidence.from_dict(payload).to_dict() == payload


def test_evidence_requires_provenance_constructed_from():
    with pytest.raises(EvidenceSchemaError, match="constructed_from"):
        Evidence(
            evidence_id="ev_bad",
            evidence_type="history",
            source_item="i1",
            target_item="i2",
            support_items=["i1"],
            provenance={"split": "train"},
        )
