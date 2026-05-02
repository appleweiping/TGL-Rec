from __future__ import annotations

import pytest

from llm4rec.evaluation.prediction_schema import PredictionSchemaError, validate_candidate_ref


def test_candidate_ref_requires_checksum_and_row_id() -> None:
    ref = {
        "artifact_id": "tiny_candidates_protocol_v1",
        "artifact_path": "outputs/artifacts/protocol_v1/tiny/candidates.jsonl",
        "artifact_sha256": "abc123",
        "candidate_row_id": "test|u1|i9",
        "candidate_size": "1000",
    }

    normalized = validate_candidate_ref(ref)

    assert normalized["candidate_size"] == 1000


def test_candidate_ref_rejects_missing_required_fields() -> None:
    with pytest.raises(PredictionSchemaError):
        validate_candidate_ref({"artifact_id": "x"})
