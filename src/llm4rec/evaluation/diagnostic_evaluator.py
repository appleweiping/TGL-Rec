"""Evaluator for method x perturbation diagnostic predictions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from llm4rec.evaluation.prediction_schema import validate_prediction_row
from llm4rec.io.artifacts import read_jsonl, write_csv_rows, write_json
from llm4rec.metrics.ranking import aggregate_ranking_metrics, coverage
from llm4rec.metrics.validity import aggregate_validity_metrics


def evaluate_diagnostic_predictions(
    *,
    prediction_rows: list[dict[str, Any]],
    item_catalog: set[str],
    ks: tuple[int, ...],
    candidate_protocol: str,
) -> list[dict[str, Any]]:
    """Return metrics grouped by method and perturbation."""

    validated = [
        validate_prediction_row(row, candidate_protocol=candidate_protocol)
        for row in prediction_rows
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        perturbation = str(row.get("metadata", {}).get("perturbation", "unknown"))
        groups[(str(row["method"]), perturbation)].append(row)
    metric_rows: list[dict[str, Any]] = []
    for (method, perturbation), rows in sorted(groups.items()):
        metrics = aggregate_ranking_metrics(rows, ks=ks)
        metrics.update(
            aggregate_validity_metrics(
                rows,
                item_catalog=item_catalog,
                candidate_protocol=candidate_protocol,
            )
        )
        metrics["coverage"] = coverage(rows, item_catalog)
        metric_rows.append(
            {
                "method": method,
                "num_predictions": len(rows),
                "perturbation": perturbation,
                **dict(sorted(metrics.items())),
            }
        )
    return metric_rows


def evaluate_prediction_file(
    *,
    predictions_path: str | Path,
    item_catalog_path: str | Path,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    ks: tuple[int, ...],
    candidate_protocol: str,
) -> list[dict[str, Any]]:
    """Evaluate a diagnostic prediction JSONL file and write JSON/CSV metrics."""

    item_catalog = {str(row["item_id"]) for row in read_jsonl(item_catalog_path)}
    rows = read_jsonl(predictions_path)
    metric_rows = evaluate_diagnostic_predictions(
        prediction_rows=rows,
        item_catalog=item_catalog,
        ks=ks,
        candidate_protocol=candidate_protocol,
    )
    write_json(output_json_path, {"metrics": metric_rows})
    write_csv_rows(output_csv_path, metric_rows)
    return metric_rows
