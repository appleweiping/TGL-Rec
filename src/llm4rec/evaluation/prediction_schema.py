"""Prediction JSONL schema validation."""

from __future__ import annotations

from typing import Any


PREDICTION_SCHEMA_V2 = "prediction_row_v2"
CANDIDATE_SCHEMA_COMPACT_REF = "compact_ref_v1"


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
    candidate_ref = row.get("candidate_ref")
    has_compact_ref = candidate_ref is not None
    if candidate_items is None:
        candidate_items = []
    if not isinstance(candidate_items, list) or not all(
        isinstance(item, str) for item in candidate_items
    ):
        raise PredictionSchemaError("candidate_items must be list[str] when present")
    if has_compact_ref:
        candidate_ref = validate_candidate_ref(candidate_ref)
    if candidate_protocol != "no_candidates" and not candidate_items and not has_compact_ref:
        raise PredictionSchemaError(
            "candidate_items may be omitted only when candidate_ref is present"
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
    if has_compact_ref:
        normalized["candidate_ref"] = candidate_ref
    normalized["scores"] = [float(score) for score in scores]
    normalized["method"] = str(row.get("method", "unknown"))
    normalized["domain"] = None if row.get("domain") is None else str(row.get("domain"))
    normalized["raw_output"] = row.get("raw_output")
    normalized["metadata"] = metadata
    normalized["schema_version"] = str(row.get("schema_version", PREDICTION_SCHEMA_V2))
    return normalized


def validate_candidate_ref(value: Any) -> dict[str, Any]:
    """Validate and normalize a compact candidate reference."""

    if not isinstance(value, dict):
        raise PredictionSchemaError("candidate_ref must be an object")
    required = {
        "artifact_id",
        "artifact_path",
        "artifact_sha256",
        "candidate_row_id",
        "candidate_size",
    }
    missing = sorted(required - set(value))
    if missing:
        raise PredictionSchemaError(f"candidate_ref missing required fields: {missing}")
    normalized = dict(value)
    for key in ("artifact_id", "artifact_path", "artifact_sha256", "candidate_row_id"):
        if normalized.get(key) in (None, ""):
            raise PredictionSchemaError(f"candidate_ref has empty field: {key}")
        normalized[key] = str(normalized[key])
    try:
        normalized["candidate_size"] = int(normalized["candidate_size"])
    except (TypeError, ValueError) as exc:
        raise PredictionSchemaError("candidate_ref.candidate_size must be int") from exc
    if normalized["candidate_size"] <= 0:
        raise PredictionSchemaError("candidate_ref.candidate_size must be positive")
    for optional in (
        "candidate_pool_artifact",
        "candidate_pool_sha256",
        "candidate_storage",
        "split",
        "target_inclusion_rule",
    ):
        if optional in normalized and normalized[optional] is not None:
            normalized[optional] = str(normalized[optional])
    return normalized
