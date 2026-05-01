"""Shared evaluator for Phase 1 prediction JSONL files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from llm4rec.evaluation.export import write_evaluation_outputs
from llm4rec.evaluation.prediction_schema import PredictionSchemaError, validate_prediction_row
from llm4rec.io.artifacts import read_jsonl
from llm4rec.metrics.ranking import aggregate_ranking_metrics, coverage
from llm4rec.metrics.validity import aggregate_validity_metrics


def evaluate_predictions(
    *,
    predictions_path: str | Path,
    item_catalog_path: str | Path,
    output_dir: str | Path,
    ks: tuple[int, ...],
    candidate_protocol: str,
) -> dict[str, Any]:
    """Validate predictions, compute metrics, and write evaluation outputs."""

    raw_rows = read_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        try:
            rows.append(validate_prediction_row(row, candidate_protocol=candidate_protocol))
        except PredictionSchemaError as exc:
            raise PredictionSchemaError(f"invalid prediction row {index}: {exc}") from exc
    if not rows:
        raise ValueError(f"No prediction rows found: {predictions_path}")

    item_rows = read_jsonl(item_catalog_path)
    item_catalog = {str(row["item_id"]) for row in item_rows}
    overall = _metrics_for_rows(rows, item_catalog=item_catalog, ks=ks, candidate_protocol=candidate_protocol)
    by_domain_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain_rows[str(row.get("domain") or "unknown")].append(row)
    by_domain = {
        domain: _metrics_for_rows(
            domain_rows,
            item_catalog=item_catalog,
            ks=ks,
            candidate_protocol=candidate_protocol,
        )
        for domain, domain_rows in sorted(by_domain_rows.items())
    }
    by_method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_rows[str(row.get("method") or "unknown")].append(row)
    by_method = {
        method: _metrics_for_rows(
            method_rows,
            item_catalog=item_catalog,
            ks=ks,
            candidate_protocol=candidate_protocol,
        )
        for method, method_rows in sorted(by_method_rows.items())
    }
    metrics = {
        "by_domain": by_domain,
        "by_method": by_method,
        "candidate_protocol": candidate_protocol,
        "num_predictions": len(rows),
        "overall": overall,
    }
    write_evaluation_outputs(output_dir, metrics)
    return metrics


def _metrics_for_rows(
    rows: list[dict[str, Any]],
    *,
    item_catalog: set[str],
    ks: tuple[int, ...],
    candidate_protocol: str,
) -> dict[str, float]:
    metrics = aggregate_ranking_metrics(rows, ks=ks)
    metrics.update(
        aggregate_validity_metrics(
            rows,
            item_catalog=item_catalog,
            candidate_protocol=candidate_protocol,
        )
    )
    metrics["coverage"] = coverage(rows, item_catalog)
    return dict(sorted(metrics.items()))
