"""Item text construction helpers."""

from __future__ import annotations

from typing import Any


def item_text(row: dict[str, Any]) -> str:
    """Build a deterministic item text string from common metadata fields."""

    parts = [
        row.get("title"),
        row.get("category"),
        row.get("brand"),
        row.get("description"),
        row.get("raw_text"),
    ]
    return " ".join(str(part).strip() for part in parts if part not in (None, ""))
