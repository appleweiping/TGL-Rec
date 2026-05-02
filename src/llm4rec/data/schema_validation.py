"""Schema normalization helpers for Amazon Reviews 2023 conversion."""

from __future__ import annotations

from typing import Any

USER_FIELDS = ["user_id", "reviewerID", "reviewer_id"]
ITEM_FIELDS = ["parent_asin", "asin", "item_id"]
TIMESTAMP_FIELDS = ["timestamp", "unixReviewTime", "unix_review_time"]
RATING_FIELDS = ["rating", "overall", "stars"]
TEXT_FIELDS = ["title", "description", "features", "categories", "main_category", "brand", "store"]


def first_present(row: dict[str, Any], fields: list[str]) -> Any:
    """Return the first non-empty row value among aliases."""

    for field in fields:
        value = row.get(field)
        if value not in (None, "", [], {}):
            return value
    return None


def normalize_interaction(row: dict[str, Any], domain: str) -> tuple[dict[str, Any] | None, str | None]:
    """Map a raw review row to the unified interaction schema."""

    user_id = first_present(row, USER_FIELDS)
    item_id = first_present(row, ITEM_FIELDS)
    timestamp = _normalize_timestamp(first_present(row, TIMESTAMP_FIELDS))
    rating = _optional_float(first_present(row, RATING_FIELDS))
    if user_id in (None, ""):
        return None, "missing_user_id"
    if item_id in (None, ""):
        return None, "missing_item_id"
    if timestamp is None:
        return None, "missing_timestamp"
    return (
        {
            "domain": domain,
            "item_id": str(item_id),
            "rating": rating,
            "timestamp": timestamp,
            "user_id": str(user_id),
        },
        None,
    )


def normalize_item(row: dict[str, Any], domain: str) -> tuple[dict[str, Any] | None, str | None]:
    """Map a raw metadata row to the unified item schema."""

    item_id = first_present(row, ITEM_FIELDS)
    if item_id in (None, ""):
        return None, "missing_item_id"
    title = _string_or_none(row.get("title"))
    description = _text_from_value(row.get("description"))
    category = _category_text(row)
    brand = _string_or_none(row.get("brand") or row.get("store"))
    raw_text = " ".join(
        part
        for part in [
            title,
            description,
            _text_from_value(row.get("features")),
            category,
            brand,
        ]
        if part
    ).strip()
    return (
        {
            "brand": brand,
            "category": category,
            "description": description,
            "domain": domain,
            "item_id": str(item_id),
            "raw_text": raw_text or None,
            "title": title,
        },
        None if raw_text or title else "missing_text",
    )


def detected_fields(rows: list[dict[str, Any]]) -> list[str]:
    """Return sorted fields seen in sampled rows."""

    fields: set[str] = set()
    for row in rows:
        fields.update(str(key) for key in row)
    return sorted(fields)


def can_convert_review_fields(fields: list[str]) -> bool:
    available = set(fields)
    return bool(available & set(USER_FIELDS)) and bool(available & set(ITEM_FIELDS)) and bool(available & set(TIMESTAMP_FIELDS))


def can_convert_item_fields(fields: list[str]) -> bool:
    available = set(fields)
    return bool(available & set(ITEM_FIELDS)) and bool(available & set(TEXT_FIELDS))


def _normalize_timestamp(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    if number > 9_999_999_999:
        number = number / 1000.0
    return int(number)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _category_text(row: dict[str, Any]) -> str | None:
    value = row.get("categories")
    if value not in (None, "", [], {}):
        text = _text_from_value(value)
        if text:
            return text
    return _string_or_none(row.get("main_category"))


def _text_from_value(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _text_from_value(item)
            if text:
                parts.append(text)
        return " ".join(parts).strip() or None
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            text = _text_from_value(value[key])
            if text:
                parts.append(text)
        return " ".join(parts).strip() or None
    return str(value).strip() or None


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None
