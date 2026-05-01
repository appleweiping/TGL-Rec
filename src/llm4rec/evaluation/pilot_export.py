"""NON_REPORTABLE pilot table export from metrics files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_csv_rows


def export_pilot_tables(run_dir: str | Path) -> dict[str, Any]:
    """Export pilot CSV/LaTeX tables from metrics JSON files only."""

    root = Path(run_dir)
    metrics_root = root / "metrics"
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        method = metrics_path.parent.name
        for method_name, values in sorted(metrics.get("by_method", {method: metrics.get("overall", {})}).items()):
            row: dict[str, Any] = {
                "NON_REPORTABLE": "NON_REPORTABLE",
                "candidate_protocol": metrics.get("candidate_protocol", "sampled_fixed"),
                "method": method_name if method_name != "unknown" else method,
                "metrics_path": str(metrics_path),
                "num_predictions": metrics.get("num_predictions", 0),
                "run_dir": str(root),
            }
            for name, value in sorted(values.items()):
                if isinstance(value, (int, float)):
                    row[name] = value
            rows.append(row)
    if (root / "ablation_manifest.json").is_file():
        csv_name = "ablation_table.csv"
        tex_name = "ablation_table.tex"
    else:
        csv_name = "pilot_table.csv"
        tex_name = "pilot_table.tex"
    write_csv_rows(root / csv_name, rows)
    (root / tex_name).write_text(_latex(rows), encoding="utf-8", newline="\n")
    return {
        "non_reportable_marker": "NON_REPORTABLE",
        "row_count": len(rows),
        "table_csv": str(root / csv_name),
        "table_tex": str(root / tex_name),
    }


def _latex(rows: list[dict[str, Any]]) -> str:
    columns = ["NON_REPORTABLE", "method", "Recall@5", "NDCG@5", "MRR@5"]
    lines = ["\\begin{tabular}{lllll}", " & ".join(columns) + " \\\\"]
    for row in rows:
        lines.append(" & ".join(str(row.get(column, "")) for column in columns) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"
