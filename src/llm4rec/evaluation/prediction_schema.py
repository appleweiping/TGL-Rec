"""Prediction JSONL schema validation."""

from __future__ import annotations

from typing import Any


class PredictionSchemaError(ValueError):
    """Raised when a prediction row violates the schema."""


def validate_prediction_row(
    row: dict[str, Any],
    *,
    candidate_protocol: str,
) -> dict[str, Any]:
    """Validate and normalize one prediction row."""

    if not isinstance(row, dict):
        raise PredictionSchemaError("prediction row must be an object")
    for field in ("user_id", "target_item"):
        if field not in row or row[field] in (None, ""):
            raise PredictionSchemaError(f"missing required field: {field}")
    predicted_items = row.get("predicted_items")
    if not isinstance(predicted_items, list) or not all(
        isinstance(item, str) for item in predicted_items
    ):
        raise PredictionSchemaError("predicted_items must be list[str]")
    candidate_items = row.get("candidate_items")
    if not isinstance(candidate_items, list) or not all(
        isinstance(item, str) for item in candidate_items
    ):
        raise PredictionSchemaError("candidate_items must be list[str]")
    if candidate_protocol != "no_candidates" and not candidate_items:
        raise PredictionSchemaError(
            "candidate_items may be empty only when candidate_protocol is no_candidates"
        )
    scores = row.get("scores", [])
    if not isinstance(scores, list):
        raise PredictionSchemaError("scores must be a list")
    if scores and len(scores) != len(predicted_items):
        raise PredictionSchemaError("scores must be empty or match predicted_items length")
    if not all(isinstance(score, (int, float)) for score in scores):
        raise PredictionSchemaError("scores must contain only numeric values")
    metadata = row.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise PredictionSchemaError("metadata must be an object")
    normalized = dict(row)
    normalized["user_id"] = str(row["user_id"])
    normalized["target_item"] = str(row["target_item"])
    normalized["predicted_items"] = [str(item) for item in predicted_items]
    normalized["candidate_items"] = [str(item) for item in candidate_items]
    normalized["scores"] = [float(score) for score in scores]
    normalized["method"] = str(row.get("method", "unknown"))
    normalized["domain"] = None if row.get("domain") is None else str(row.get("domain"))
    normalized["raw_output"] = row.get("raw_output")
    normalized["metadata"] = metadata
    return normalized
