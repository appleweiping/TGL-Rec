"""Data contracts for the Phase 1 LLM4Rec skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataSchemaError(ValueError):
    """Raised when input data violates the expected schema."""


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: int | float | None
    rating: float | None
    domain: str | None


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    title: str
    description: str | None
    category: str | None
    brand: str | None
    domain: str | None
    raw_text: str | None


@dataclass(frozen=True)
class UserExample:
    user_id: str
    history: list[str]
    target: str
    candidates: list[str] | None
    domain: str | None


@dataclass(frozen=True)
class PreprocessResult:
    output_dir: Path
    metadata: dict[str, Any]


REQUIRED_INTERACTION_FIELDS = {"user_id", "item_id", "timestamp", "rating", "domain"}
REQUIRED_ITEM_FIELDS = {
    "item_id",
    "title",
    "description",
    "category",
    "brand",
    "domain",
    "raw_text",
}


def require_fields(row: dict[str, Any], required: set[str], *, label: str) -> None:
    """Validate required keys in a row."""

    missing = sorted(required - set(row))
    if missing:
        raise DataSchemaError(f"{label} is missing required fields: {missing}")
