from __future__ import annotations

import pytest

from llm4rec.evaluation.prediction_schema import (
    PredictionSchemaError,
    validate_prediction_row,
)


def test_prediction_schema_accepts_valid_row_with_duplicates() -> None:
    row = {
        "user_id": "u1",
        "target_item": "i3",
        "candidate_items": ["i1", "i3"],
        "predicted_items": ["i3", "i3", "i1"],
        "scores": [1.0, 0.9, 0.1],
        "method": "skeleton",
        "domain": "tiny",
        "raw_output": None,
        "metadata": {},
    }
    normalized = validate_prediction_row(row, candidate_protocol="full_catalog")
    assert normalized["predicted_items"] == ["i3", "i3", "i1"]


def test_prediction_schema_rejects_bad_scores_and_missing_candidates() -> None:
    row = {
        "user_id": "u1",
        "target_item": "i3",
        "candidate_items": ["i1", "i3"],
        "predicted_items": ["i3"],
        "scores": [1.0, 0.5],
    }
    with pytest.raises(PredictionSchemaError):
        validate_prediction_row(row, candidate_protocol="full_catalog")
    row["scores"] = [1.0]
    row["candidate_items"] = []
    with pytest.raises(PredictionSchemaError):
        validate_prediction_row(row, candidate_protocol="full_catalog")


def test_no_candidates_protocol_allows_empty_candidates() -> None:
    row = {
        "user_id": "u1",
        "target_item": "i3",
        "candidate_items": [],
        "predicted_items": ["i3"],
        "scores": [],
    }
    assert validate_prediction_row(row, candidate_protocol="no_candidates")["candidate_items"] == []


def test_prediction_schema_accepts_compact_candidate_ref() -> None:
    row = {
        "user_id": "u1",
        "target_item": "i3",
        "candidate_ref": {
            "artifact_id": "tiny_candidates_protocol_v1",
            "artifact_path": "outputs/artifacts/protocol_v1/tiny/candidates.jsonl",
            "artifact_sha256": "abc123",
            "candidate_row_id": "test|u1|i3",
            "candidate_size": 2,
        },
        "predicted_items": ["i3"],
        "scores": [1.0],
        "metadata": {"candidate_schema": "compact_ref_v1"},
    }

    normalized = validate_prediction_row(row, candidate_protocol="fixed_sampled")

    assert normalized["candidate_items"] == []
    assert normalized["candidate_ref"]["candidate_row_id"] == "test|u1|i3"
