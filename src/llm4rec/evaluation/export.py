"""Evaluation export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_csv_rows, write_json, write_metric_csv


def flatten_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten overall, per-domain, and per-method metrics to long CSV rows."""

    rows: list[dict[str, Any]] = []
    for name, value in sorted(metrics.get("overall", {}).items()):
        if isinstance(value, (int, float)):
            rows.append({"scope": "overall", "domain": "ALL", "metric": name, "value": value})
    for domain, domain_metrics in sorted(metrics.get("by_domain", {}).items()):
        for name, value in sorted(domain_metrics.items()):
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "scope": "domain",
                        "domain": domain,
                        "metric": name,
                        "value": value,
                    }
                )
    for method, method_metrics in sorted(metrics.get("by_method", {}).items()):
        for name, value in sorted(method_metrics.items()):
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "scope": f"method:{method}",
                        "domain": "ALL",
                        "metric": name,
                        "value": value,
                    }
                )
    return rows


def write_evaluation_outputs(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    """Write metrics.json and metrics.csv."""

    root = Path(output_dir)
    write_json(root / "metrics.json", metrics)
    write_metric_csv(root / "metrics.csv", flatten_metrics(metrics))


def collect_metric_files(input_dir: str | Path) -> list[Path]:
    """Collect metrics.json files below an output root."""

    root = Path(input_dir)
    if root.is_file() and root.name == "metrics.json":
        return [root]
    return sorted(path for path in root.rglob("metrics.json") if path.is_file())


def export_paper_tables(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Export CSV and LaTeX-ready tables from metrics files only."""

    import json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for metrics_path in collect_metric_files(input_dir):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        for method, values in sorted(metrics.get("by_method", {"ALL": metrics.get("overall", {})}).items()):
            row = {
                "candidate_protocol": metrics.get("candidate_protocol", ""),
                "dataset": metrics.get("dataset", "unknown"),
                "method": method,
                "run_dir": str(run_dir),
                "seed": metrics.get("seed", ""),
                "split": metrics.get("split_strategy", ""),
            }
            for name, value in sorted(values.items()):
                if isinstance(value, (int, float)):
                    row[name] = value
            rows.append(row)
    write_csv_rows(output / "paper_table.csv", rows)
    latex_path = output / "paper_table.tex"
    latex_path.write_text(_latex_table(rows), encoding="utf-8", newline="\n")
    manifest = {
        "input_dir": str(Path(input_dir)),
        "metric_file_count": len(collect_metric_files(input_dir)),
        "outputs": ["paper_table.csv", "paper_table.tex"],
        "source": "metrics_files_only",
    }
    write_json(output / "table_manifest.json", manifest)
    return manifest


def _latex_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "\\begin{tabular}{ll}\nmethod & note \\\\\n\\end{tabular}\n"
    columns = ["dataset", "method", "candidate_protocol", "Recall@5", "NDCG@5", "MRR@5"]
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(columns) + " \\\\"]
    for row in rows:
        lines.append(" & ".join(str(row.get(column, "")) for column in columns) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"
