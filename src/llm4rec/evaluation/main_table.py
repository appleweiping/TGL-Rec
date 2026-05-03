"""Export protocol-v1 multi-seed main accuracy tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import write_csv_rows


TABLE_LABEL = "PAPER-SCALE MULTI-SEED MAIN ACCURACY RESULTS, PROTOCOL_V1"
TABLE_METRICS = [
    ("Recall@5", "Recall@5 mean±std"),
    ("NDCG@5", "NDCG@5 mean±std"),
    ("MRR@10", "MRR@10 mean±std"),
    ("coverage", "coverage mean±std"),
    ("novelty", "novelty mean±std"),
    ("long_tail_ratio", "long_tail_ratio mean±std"),
    ("runtime_seconds", "runtime mean±std"),
]


def export_main_accuracy_multiseed_tables(run_dir: str | Path) -> dict[str, Any]:
    """Write CSV and LaTeX mean/std tables for Phase 9C."""

    root = resolve_path(run_dir)
    aggregate_rows = _read_csv(root / "aggregate_metrics.csv")
    significance_rows = _read_csv(root / "significance_tests.csv")
    table_rows = build_main_accuracy_table_rows(aggregate_rows, significance_rows)
    columns = [
        "dataset",
        "method",
        *[label for _metric, label in TABLE_METRICS],
        "significance marker if available",
    ]
    csv_path = root / "table_main_accuracy_mean_std.csv"
    tex_path = root / "table_main_accuracy_mean_std.tex"
    write_csv_rows(csv_path, table_rows, fieldnames=columns)
    tex_path.write_text(_latex_table(table_rows, columns), encoding="utf-8", newline="\n")
    return {"row_count": len(table_rows), "table_csv": str(csv_path), "table_tex": str(tex_path)}


def build_main_accuracy_table_rows(
    aggregate_rows: list[dict[str, Any]],
    significance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build formatted mean/std rows from long aggregate metrics."""

    index = {
        (str(row.get("dataset", "")), str(row.get("method", "")), str(row.get("metric", ""))): row
        for row in aggregate_rows
    }
    keys = sorted({(dataset, method) for dataset, method, _metric in index})
    markers = _significance_markers(significance_rows)
    rows: list[dict[str, Any]] = []
    for dataset, method in keys:
        row: dict[str, Any] = {"dataset": dataset, "method": method}
        for metric, label in TABLE_METRICS:
            metric_row = index.get((dataset, method, metric), {})
            row[label] = _format_mean_std(metric_row.get("mean", 0.0), metric_row.get("std", 0.0))
        row["significance marker if available"] = markers.get((dataset, method), "")
        rows.append(row)
    return rows


def _significance_markers(significance_rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    markers: dict[tuple[str, str], str] = {}
    for row in significance_rows:
        if str(row.get("metric", "")) != "Recall@5":
            continue
        if str(row.get("significant_at_0_05", "")).lower() not in {"true", "1"}:
            continue
        if str(row.get("effect_direction", "")) != "method_a_better":
            continue
        notes = str(row.get("notes", ""))
        if "best_non_ours" not in notes:
            continue
        markers[(str(row.get("dataset", "")), str(row.get("method_a", "")))] = "*"
    return markers


def _format_mean_std(mean: Any, std: Any) -> str:
    return f"{float(mean or 0.0):.6f}±{float(std or 0.0):.6f}"


def _latex_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{PAPER-SCALE MULTI-SEED MAIN ACCURACY RESULTS, PROTOCOL\\_V1}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        " & ".join(_escape_latex(column) for column in columns) + " \\\\",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(str(row.get(column, ""))) for column in columns) + " \\\\")
    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _escape_latex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
