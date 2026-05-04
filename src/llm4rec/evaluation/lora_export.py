"""Separate LoRA result table export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import write_csv_rows


def export_lora_table(run_dir: str | Path) -> dict[str, str]:
    """Export a separate LoRA comparison table."""

    root = resolve_path(run_dir)
    rows = _read_csv(root / "metrics" / "metrics_by_method.csv") if (root / "metrics" / "metrics_by_method.csv").is_file() else []
    rows.extend(_phase9c_baselines())
    rows.extend(_phase9d_best_api())
    columns = ["dataset", "method", "Recall@5", "NDCG@5", "MRR@10", "validity_rate", "hallucination_rate", "parse_success_rate"]
    csv_path = root / "table_lora_8b.csv"
    tex_path = root / "table_lora_8b.tex"
    write_csv_rows(csv_path, rows, fieldnames=columns)
    tex_path.write_text(_latex(rows, columns), encoding="utf-8", newline="\n")
    return {"table_csv": str(csv_path), "table_tex": str(tex_path)}


def _phase9c_baselines() -> list[dict[str, Any]]:
    path = resolve_path("outputs/paper_runs/protocol_v1/main_accuracy_multiseed/aggregate_metrics.csv")
    if not path.is_file():
        return []
    rows = _read_csv(path)
    output: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        best = _best_method(dataset_rows)
        for method in {best, "time_graph_evidence"} - {""}:
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "Recall@5": _metric(dataset_rows, method, "Recall@5"),
                    "NDCG@5": _metric(dataset_rows, method, "NDCG@5"),
                    "MRR@10": _metric(dataset_rows, method, "MRR@10"),
                }
            )
    return output


def _phase9d_best_api() -> list[dict[str, Any]]:
    path = resolve_path("outputs/paper_runs/protocol_v1/deepseek_llm/full/metrics_by_method.csv")
    if not path.is_file():
        return []
    rows = _read_csv(path)
    output: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        best = max(dataset_rows, key=lambda row: float(row.get("NDCG@5", 0.0) or 0.0))
        best["method"] = f"deepseek_best_api::{best['method']}"
        output.append(best)
    return output


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _best_method(rows: list[dict[str, str]]) -> str:
    recall_rows = [row for row in rows if row.get("metric") == "Recall@5" and row.get("method") != "time_graph_evidence"]
    return str(max(recall_rows, key=lambda row: float(row.get("mean", 0.0) or 0.0)).get("method", "")) if recall_rows else ""


def _metric(rows: list[dict[str, str]], method: str, metric: str) -> str:
    for row in rows:
        if row.get("method") == method and row.get("metric") == metric:
            return str(row.get("mean", ""))
    return ""


def _latex(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(columns) + " \\\\"]
    for row in rows:
        lines.append(" & ".join(str(row.get(column, "")).replace("_", "\\_") for column in columns) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"
