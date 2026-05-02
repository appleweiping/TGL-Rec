"""Candidate set construction."""

from __future__ import annotations

import hashlib
import random
from typing import Any


def build_candidate_sets(
    labeled_interactions: list[dict[str, Any]],
    item_ids: list[str],
    *,
    protocol: str,
    candidate_size: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic candidate rows for validation and test examples."""

    catalog = sorted({str(item_id) for item_id in item_ids})
    rows: list[dict[str, Any]] = []
    for row in labeled_interactions:
        if row["split"] not in {"valid", "test"}:
            continue
        target = str(row["item_id"])
        candidates = _build_one_candidate_set(
            catalog,
            target,
            protocol=protocol,
            candidate_size=candidate_size,
            seed=seed,
            row_key=f"{row['split']}|{row['user_id']}|{target}",
        )
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


def _build_one_candidate_set(
    catalog: list[str],
    target: str,
    *,
    protocol: str,
    candidate_size: int | None,
    seed: int,
    row_key: str,
) -> list[str]:
    if target not in catalog:
        catalog = sorted([*catalog, target])
    if protocol == "full_catalog":
        return list(catalog)
    if protocol not in {"fixed_sampled", "fixed_shared_candidates", "sampled_fixed"}:
        raise ValueError(f"Unsupported candidate protocol: {protocol!r}")
    size = min(int(candidate_size or len(catalog)), len(catalog))
    negatives_needed = max(0, size - 1)
    if negatives_needed >= len(catalog) - 1:
        negatives = [item for item in catalog if item != target]
    else:
        digest = hashlib.sha256(f"{seed}|{row_key}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
        chosen: set[str] = set()
        while len(chosen) < negatives_needed:
            item = catalog[rng.randrange(len(catalog))]
            if item != target:
                chosen.add(item)
        negatives = sorted(chosen)
    candidates = sorted([*negatives, target])
    if target not in candidates:
        raise ValueError(f"target missing from candidate set for {row_key}")
    return candidates
