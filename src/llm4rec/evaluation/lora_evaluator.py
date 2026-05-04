"""Evaluation helpers for local LoRA reranking outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import read_jsonl, write_csv_rows, write_json
from llm4rec.metrics.ranking import aggregate_ranking_metrics


def evaluate_lora_predictions(predictions_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate local LoRA prediction JSONL with ranking and validity metrics."""

    rows = read_jsonl(predictions_path)
    invalid = 0
    total = 0
    parse_success = 0
    target_in_top_m = 0
    for row in rows:
        candidates = {str(item) for item in row.get("candidate_items", [])}
        if row.get("target_item") in candidates:
            target_in_top_m += 1
        if row.get("metadata", {}).get("parse_success", False):
            parse_success += 1
        for item in row.get("predicted_items", []):
            total += 1
            if str(item) not in candidates:
                invalid += 1
    metrics = {
        **aggregate_ranking_metrics(rows, ks=(1, 5, 10)),
        "candidate_adherence_rate": 1.0 - invalid / float(total or 1),
        "hallucination_rate": invalid / float(total or 1),
        "parse_success_rate": parse_success / float(len(rows) or 1),
        "target_in_top_m_rate": target_in_top_m / float(len(rows) or 1),
        "validity_rate": 1.0 - invalid / float(total or 1),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    write_csv_rows(out / "metrics.csv", [{"metric": key, "value": value} for key, value in sorted(metrics.items())])
    return metrics
