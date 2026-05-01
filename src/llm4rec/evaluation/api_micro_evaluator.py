"""Evaluation for Phase 3B API micro diagnostics."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from llm4rec.diagnostics.llm_grounding import aggregate_grounding
from llm4rec.diagnostics.statistics import compare_prediction_sets
from llm4rec.evaluation.prediction_schema import validate_prediction_row
from llm4rec.metrics.ranking import aggregate_ranking_metrics
from llm4rec.metrics.validity import aggregate_validity_metrics


def evaluate_api_micro_predictions(
    *,
    prediction_rows: list[dict[str, Any]],
    item_catalog: set[str],
    ks: tuple[int, ...],
    candidate_protocol: str,
    reference_variant: str = "history_only",
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate per prompt variant and per case group."""

    if not prediction_rows:
        return [], [], [], []
    validated = [
        validate_prediction_row(row, candidate_protocol=candidate_protocol)
        for row in prediction_rows
    ]
    metric_rows: list[dict[str, Any]] = []
    for case_group, prompt_variant, rows in _group_predictions(validated):
        metrics = aggregate_ranking_metrics(rows, ks=ks)
        metrics.update(
            aggregate_validity_metrics(
                rows,
                item_catalog=item_catalog,
                candidate_protocol=candidate_protocol,
            )
        )
        metrics.update(_parse_metrics(rows))
        metrics.update(_usage_metrics(rows))
        metrics.update(aggregate_grounding(rows))
        metric_rows.append(
            {
                "case_group": case_group,
                "method": rows[0].get("method", "api_micro_llm_rerank"),
                "num_predictions": len(rows),
                "prompt_variant": prompt_variant,
                **dict(sorted(metrics.items())),
            }
        )
    delta_rows = _metric_deltas(metric_rows, reference_variant=reference_variant)
    overlap_rows = _prompt_overlaps(validated, reference_variant=reference_variant, top_k=top_k)
    comparison_rows = _case_level_comparison(
        validated, reference_variant=reference_variant, top_k=top_k
    )
    return metric_rows, delta_rows, overlap_rows, comparison_rows


def _group_predictions(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        prompt_variant = str(metadata.get("prompt_variant", "unknown"))
        case_group = str(metadata.get("case_group", metadata.get("sample_group", "unknown")))
        groups[(case_group, prompt_variant)].append(row)
        groups[("ALL", prompt_variant)].append(row)
    return [
        (case_group, prompt_variant, group_rows)
        for (case_group, prompt_variant), group_rows in sorted(groups.items())
    ]


def _parse_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = float(len(rows) or 1)
    parse_success = sum(1.0 for row in rows if row.get("metadata", {}).get("parse_success"))
    candidate_adherent = sum(
        1.0 for row in rows if not row.get("metadata", {}).get("invalid_item_ids")
    )
    return {
        "candidate_adherence_rate": candidate_adherent / total,
        "parse_success_rate": parse_success / total,
    }


def _usage_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "cache_hit_rate": 0.0,
            "completion_tokens": 0.0,
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "prompt_tokens": 0.0,
            "total_tokens": 0.0,
        }
    usages = [row.get("metadata", {}).get("llm_usage", {}) for row in rows]
    latencies = sorted(float(usage.get("latency_ms", 0.0) or 0.0) for usage in usages)
    return {
        "cache_hit_rate": sum(1.0 for usage in usages if usage.get("cache_hit"))
        / float(len(usages)),
        "completion_tokens": sum(
            float(usage.get("completion_tokens", 0.0) or 0.0) for usage in usages
        ),
        "mean_latency_ms": sum(latencies) / float(len(latencies)),
        "p50_latency_ms": float(median(latencies)),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "prompt_tokens": sum(float(usage.get("prompt_tokens", 0.0) or 0.0) for usage in usages),
        "total_tokens": sum(float(usage.get("total_tokens", 0.0) or 0.0) for usage in usages),
    }


def _metric_deltas(rows: list[dict[str, Any]], *, reference_variant: str) -> list[dict[str, Any]]:
    by_group = defaultdict(dict)
    for row in rows:
        by_group[str(row["case_group"])][str(row["prompt_variant"])] = row
    output: list[dict[str, Any]] = []
    for case_group, variants in sorted(by_group.items()):
        base = variants.get(reference_variant)
        if base is None:
            continue
        for prompt_variant, row in sorted(variants.items()):
            entry = {
                "case_group": case_group,
                "prompt_variant": prompt_variant,
                "reference_variant": reference_variant,
            }
            for metric in ("Recall@5", "NDCG@5", "MRR@5", "validity_rate", "hallucination_rate"):
                if metric in row and metric in base:
                    entry[f"delta_{metric}_vs_{reference_variant}"] = float(row[metric]) - float(
                        base[metric]
                    )
            output.append(entry)
    return output


def _prompt_overlaps(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str,
    top_k: int,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(
            row.get("metadata", {}).get(
                "case_group", row.get("metadata", {}).get("sample_group", "unknown")
            )
        )
        by_group[group].append(row)
        by_group["ALL"].append(row)
    output: list[dict[str, Any]] = []
    for case_group, group_rows in sorted(by_group.items()):
        reference = [
            row
            for row in group_rows
            if row.get("metadata", {}).get("prompt_variant") == reference_variant
        ]
        variants = sorted(
            {str(row.get("metadata", {}).get("prompt_variant", "unknown")) for row in group_rows}
        )
        for variant in variants:
            if variant == reference_variant:
                continue
            variant_rows = [
                row
                for row in group_rows
                if row.get("metadata", {}).get("prompt_variant") == variant
            ]
            stats = compare_prediction_sets(reference, variant_rows, k=top_k)
            output.append(
                {
                    "case_group": case_group,
                    "output_change_rate_vs_history_only": 1.0
                    - float(stats["mean_prediction_overlap"]),
                    "prediction_overlap_vs_history_only@K": stats["mean_prediction_overlap"],
                    "prompt_variant": variant,
                    **stats,
                }
            )
    return output


def _case_level_comparison(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str,
    top_k: int,
) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        metadata = row.get("metadata", {})
        key = (
            str(metadata.get("sample_id")),
            str(row.get("user_id")),
            str(row.get("target_item")),
        )
        by_case[key][str(metadata.get("prompt_variant", "unknown"))] = row
    output: list[dict[str, Any]] = []
    for (sample_id, user_id, target_item), variants in sorted(by_case.items()):
        base = variants.get(reference_variant)
        if base is None:
            continue
        for variant, row in sorted(variants.items()):
            if variant == reference_variant:
                continue
            stats = compare_prediction_sets([base], [row], k=top_k)
            output.append(
                {
                    "case_group": row.get("metadata", {}).get("case_group"),
                    "output_changed": float(stats["mean_prediction_overlap"]) < 1.0,
                    "prediction_overlap_vs_history_only@K": stats["mean_prediction_overlap"],
                    "prompt_variant": variant,
                    "reference_variant": reference_variant,
                    "sample_id": sample_id,
                    "target_item": target_item,
                    "user_id": user_id,
                }
            )
    return output


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)
