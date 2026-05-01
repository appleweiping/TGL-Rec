"""Comparison-table helpers for diagnostics."""

from __future__ import annotations

from typing import Any


def best_rows_by_metric(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    metric: str,
) -> list[dict[str, Any]]:
    """Return the best row per group for a metric."""

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = str(row.get(group_key, "unknown"))
        if metric not in row:
            continue
        if group not in best or float(row[metric]) > float(best[group][metric]):
            best[group] = row
    return [best[key] for key in sorted(best)]
