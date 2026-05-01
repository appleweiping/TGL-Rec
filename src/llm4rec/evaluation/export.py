"""Evaluation export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json, write_metric_csv


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
