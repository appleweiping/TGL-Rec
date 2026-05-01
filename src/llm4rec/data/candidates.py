"""Candidate set construction."""

from __future__ import annotations

from typing import Any


def build_candidate_sets(
    labeled_interactions: list[dict[str, Any]],
    item_ids: list[str],
    *,
    protocol: str,
) -> list[dict[str, Any]]:
    """Build deterministic candidate rows for validation and test examples."""

    if protocol != "full_catalog":
        raise ValueError(f"Phase 1 supports only full_catalog candidates, got {protocol!r}")
    catalog = sorted({str(item_id) for item_id in item_ids})
    rows: list[dict[str, Any]] = []
    for row in labeled_interactions:
        if row["split"] not in {"valid", "test"}:
            continue
        target = str(row["item_id"])
        candidates = list(catalog)
        if target not in candidates:
            candidates.append(target)
            candidates = sorted(candidates)
        rows.append(
            {
                "user_id": str(row["user_id"]),
                "target_item": target,
                "split": row["split"],
                "candidate_items": candidates,
                "domain": row.get("domain"),
            }
        )
    return sorted(rows, key=lambda value: (value["split"], value["user_id"], value["target_item"]))
