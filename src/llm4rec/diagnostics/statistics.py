"""Statistics helpers for diagnostic comparisons."""

from __future__ import annotations

from statistics import fmean
from typing import Any


def target_rank(predicted_items: list[str], target_item: str) -> int | None:
    """Return 1-based target rank after deterministic duplicate removal."""

    seen: set[str] = set()
    for index, item in enumerate(predicted_items, start=1):
        item_id = str(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        if item_id == str(target_item):
            return index
    return None


def prediction_overlap_at_k(left: list[str], right: list[str], *, k: int) -> float:
    """Jaccard overlap between top-k predictions."""

    left_set = set(_dedupe(left)[:k])
    right_set = set(_dedupe(right)[:k])
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / float(len(union))


def rank_correlation(left: list[str], right: list[str], *, k: int) -> float | None:
    """Spearman-like rank correlation on common top-k items."""

    left_rank = {item: index for index, item in enumerate(_dedupe(left)[:k], start=1)}
    right_rank = {item: index for index, item in enumerate(_dedupe(right)[:k], start=1)}
    common = sorted(set(left_rank) & set(right_rank))
    if len(common) < 2:
        return None
    left_values = [left_rank[item] for item in common]
    right_values = [right_rank[item] for item in common]
    return _pearson(left_values, right_values)


def compute_metric_deltas(
    metric_rows: list[dict[str, Any]],
    *,
    baseline_variant: str = "original",
    metrics: tuple[str, ...] = ("Recall@5", "NDCG@5"),
) -> list[dict[str, Any]]:
    """Compute deltas for every method/variant against original."""

    base: dict[str, dict[str, float]] = {}
    for row in metric_rows:
        if row["perturbation"] == baseline_variant:
            base[str(row["method"])] = {
                metric: float(row.get(metric, 0.0))
                for metric in metrics
                if row.get(metric) not in (None, "")
            }
    deltas: list[dict[str, Any]] = []
    for row in metric_rows:
        method = str(row["method"])
        if method not in base:
            continue
        output = {"method": method, "perturbation": row["perturbation"]}
        for metric in metrics:
            if metric in base[method] and row.get(metric) not in (None, ""):
                output[f"delta_{metric}_vs_original"] = float(row[metric]) - base[method][metric]
        deltas.append(output)
    return deltas


def compare_prediction_sets(
    original_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, float]:
    """Compare variant predictions against original by overlap and rank shift."""

    original_by_key = {
        (str(row["method"]), str(row["user_id"]), str(row["target_item"])): row
        for row in original_rows
    }
    overlaps: list[float] = []
    shifts: list[float] = []
    correlations: list[float] = []
    for row in variant_rows:
        key = (str(row["method"]), str(row["user_id"]), str(row["target_item"]))
        original = original_by_key.get(key)
        if original is None:
            continue
        overlap = prediction_overlap_at_k(
            original.get("predicted_items", []),
            row.get("predicted_items", []),
            k=k,
        )
        overlaps.append(overlap)
        original_rank = target_rank(original.get("predicted_items", []), original["target_item"])
        variant_rank = target_rank(row.get("predicted_items", []), row["target_item"])
        if original_rank is not None and variant_rank is not None:
            shifts.append(float(variant_rank - original_rank))
        corr = rank_correlation(
            original.get("predicted_items", []),
            row.get("predicted_items", []),
            k=k,
        )
        if corr is not None:
            correlations.append(float(corr))
    return {
        "mean_prediction_overlap": fmean(overlaps) if overlaps else 0.0,
        "mean_rank_correlation": fmean(correlations) if correlations else 0.0,
        "mean_target_rank_shift": fmean(shifts) if shifts else 0.0,
        "num_compared": float(len(overlaps)),
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        item_id = str(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        output.append(item_id)
    return output


def _pearson(left: list[int], right: list[int]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_den = sum((a - left_mean) ** 2 for a in left)
    right_den = sum((b - right_mean) ** 2 for b in right)
    denominator = (left_den * right_den) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator
