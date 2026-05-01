"""Serializable evidence contracts for TimeGraphEvidenceRec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EVIDENCE_TYPES = {
    "history",
    "transition",
    "time_window",
    "time_gap",
    "semantic",
    "contrastive",
    "user_drift",
}

ALLOWED_CONSTRUCTED_FROM = {"train_only", "diagnostic_only"}


class EvidenceSchemaError(ValueError):
    """Raised when an evidence object violates the schema."""


@dataclass(frozen=True)
class Evidence:
    """Grounded evidence row passed from graph retrieval to rankers or prompts."""

    evidence_id: str
    evidence_type: str
    source_item: str | None
    target_item: str | None
    support_items: list[str]
    timestamp_info: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise EvidenceSchemaError("evidence_id is required")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise EvidenceSchemaError(f"unsupported evidence_type: {self.evidence_type}")
        if not isinstance(self.support_items, list) or not all(
            isinstance(item, str) for item in self.support_items
        ):
            raise EvidenceSchemaError("support_items must be list[str]")
        if not isinstance(self.timestamp_info, dict):
            raise EvidenceSchemaError("timestamp_info must be an object")
        if not isinstance(self.stats, dict):
            raise EvidenceSchemaError("stats must be an object")
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise EvidenceSchemaError("provenance is required")
        constructed_from = self.provenance.get("constructed_from")
        if constructed_from not in ALLOWED_CONSTRUCTED_FROM:
            raise EvidenceSchemaError(
                "provenance.constructed_from must be train_only or diagnostic_only"
            )
        _assert_jsonable(self.to_dict(validate=False))

    def to_dict(self, *, validate: bool = True) -> dict[str, Any]:
        """Return a stable JSON-serializable dictionary."""

        row = {
            "evidence_id": str(self.evidence_id),
            "evidence_type": str(self.evidence_type),
            "source_item": None if self.source_item is None else str(self.source_item),
            "target_item": None if self.target_item is None else str(self.target_item),
            "support_items": [str(item) for item in self.support_items],
            "timestamp_info": _jsonable_dict(self.timestamp_info),
            "stats": _jsonable_dict(self.stats),
            "text": str(self.text),
            "provenance": _jsonable_dict(self.provenance),
            "metadata": _jsonable_dict(self.metadata),
        }
        if validate:
            _assert_jsonable(row)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Evidence":
        """Create evidence from a serialized dictionary."""

        if not isinstance(row, dict):
            raise EvidenceSchemaError("evidence row must be an object")
        return cls(
            evidence_id=str(row.get("evidence_id", "")),
            evidence_type=str(row.get("evidence_type", "")),
            source_item=None if row.get("source_item") is None else str(row.get("source_item")),
            target_item=None if row.get("target_item") is None else str(row.get("target_item")),
            support_items=[str(item) for item in row.get("support_items", [])],
            timestamp_info=dict(row.get("timestamp_info", {})),
            stats=dict(row.get("stats", {})),
            text=str(row.get("text", "")),
            provenance=dict(row.get("provenance", {})),
            metadata=dict(row.get("metadata", {})),
        )


def _jsonable_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in values.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return _jsonable_dict(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _assert_jsonable(value: Any) -> None:
    import json

    try:
        json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError as exc:  # pragma: no cover - _jsonable should prevent this.
        raise EvidenceSchemaError(f"evidence is not JSON serializable: {exc}") from exc
