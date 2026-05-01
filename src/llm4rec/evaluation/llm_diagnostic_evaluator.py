"""Evaluation utilities for Phase 3A LLM prompt diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm4rec.diagnostics.llm_grounding import aggregate_grounding
from llm4rec.diagnostics.statistics import compare_prediction_sets
from llm4rec.evaluation.prediction_schema import validate_prediction_row
from llm4rec.metrics.ranking import aggregate_ranking_metrics, coverage
from llm4rec.metrics.validity import aggregate_validity_metrics


def evaluate_llm_diagnostic_predictions(
    *,
    prediction_rows: list[dict[str, Any]],
    item_catalog: set[str],
    ks: tuple[int, ...],
    candidate_protocol: str,
    reference_variant: str = "history_only",
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate metrics, deltas, and prompt-overlap rows."""

    validated = [
        validate_prediction_row(row, candidate_protocol=candidate_protocol)
        for row in prediction_rows
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        groups[str(row.get("metadata", {}).get("prompt_variant", "unknown"))].append(row)
    metric_rows: list[dict[str, Any]] = []
    for prompt_variant, rows in sorted(groups.items()):
        metrics = aggregate_ranking_metrics(rows, ks=ks)
        metrics.update(
            aggregate_validity_metrics(
                rows,
                item_catalog=item_catalog,
                candidate_protocol=candidate_protocol,
            )
        )
        metrics["coverage"] = coverage(rows, item_catalog)
        metrics.update(_parse_metrics(rows))
        metrics.update(_usage_metrics(rows))
        metrics.update(aggregate_grounding(rows))
        metric_rows.append(
            {
                "method": rows[0].get("method", "llm_rerank_diagnostic"),
                "num_predictions": len(rows),
                "prompt_variant": prompt_variant,
                **dict(sorted(metrics.items())),
            }
        )
    deltas = _metric_deltas(metric_rows, reference_variant=reference_variant)
    overlaps = _prompt_overlaps(validated, reference_variant=reference_variant, top_k=top_k)
    return metric_rows, deltas, overlaps


def _parse_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = float(len(rows) or 1)
    parse_success = sum(1.0 for row in rows if row.get("metadata", {}).get("parse_success"))
    candidate_adherent = sum(
        1.0
        for row in rows
        if not row.get("metadata", {}).get("invalid_item_ids")
    )
    return {
        "candidate_adherence_rate": candidate_adherent / total,
        "parse_success_rate": parse_success / total,
    }


def _usage_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"completion_tokens": 0.0, "mean_latency_ms": 0.0, "prompt_tokens": 0.0, "total_tokens": 0.0}
    latencies = [
        float(row.get("metadata", {}).get("llm_usage", {}).get("latency_ms", 0.0))
        for row in rows
    ]
    return {
        "completion_tokens": sum(
            float(row.get("metadata", {}).get("llm_usage", {}).get("completion_tokens", 0.0))
            for row in rows
        ),
        "mean_latency_ms": sum(latencies) / float(len(latencies)),
        "prompt_tokens": sum(
            float(row.get("metadata", {}).get("llm_usage", {}).get("prompt_tokens", 0.0))
            for row in rows
        ),
        "total_tokens": sum(
            float(row.get("metadata", {}).get("llm_usage", {}).get("total_tokens", 0.0))
            for row in rows
        ),
    }


def _metric_deltas(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str,
) -> list[dict[str, Any]]:
    base = next((row for row in rows if row["prompt_variant"] == reference_variant), None)
    if base is None:
        return []
    output: list[dict[str, Any]] = []
    for row in rows:
        entry = {"prompt_variant": row["prompt_variant"], "reference_variant": reference_variant}
        for metric in ("Recall@5", "NDCG@5", "MRR@5", "validity_rate", "hallucination_rate"):
            if metric in row and metric in base:
                entry[f"delta_{metric}_vs_{reference_variant}"] = float(row[metric]) - float(base[metric])
        output.append(entry)
    return output


def _prompt_overlaps(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str,
    top_k: int,
) -> list[dict[str, Any]]:
    reference = [
        row
        for row in rows
        if row.get("metadata", {}).get("prompt_variant") == reference_variant
    ]
    variants = sorted({str(row.get("metadata", {}).get("prompt_variant", "unknown")) for row in rows})
    output: list[dict[str, Any]] = []
    for variant in variants:
        if variant == reference_variant:
            continue
        variant_rows = [
            row
            for row in rows
            if row.get("metadata", {}).get("prompt_variant") == variant
        ]
        stats = compare_prediction_sets(reference, variant_rows, k=top_k)
        output.append(
            {
                "output_change_rate": 1.0 - float(stats["mean_prediction_overlap"]),
                "prediction_overlap_vs_history_only@K": stats["mean_prediction_overlap"],
                "prompt_variant": variant,
                **stats,
            }
        )
    return output

