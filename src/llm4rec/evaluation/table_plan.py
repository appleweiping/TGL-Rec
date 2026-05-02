"""Paper table shells and export plan without metric values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json


def plan_paper_tables(
    manifest: dict[str, Any],
    output_path: str | Path = "outputs/launch/paper_v1/table_plan.json",
) -> dict[str, Any]:
    """Save table shells with inputs/grouping/metrics but no numbers."""

    output = resolve_path(output_path)
    ensure_dir(output.parent)
    tables = [
        _table("main_accuracy", manifest, ["Recall@10", "NDCG@10", "MRR@10"], ["dataset", "method", "seed"]),
        _table("ablation", manifest, ["Recall@10", "NDCG@10"], ["dataset", "ablation", "seed"]),
        _table("long_tail", manifest, ["long_tail_ratio", "Recall@10"], ["dataset", "method", "popularity_bucket"]),
        _table("cold_start", manifest, ["Recall@10", "NDCG@10"], ["dataset", "method", "user_sparsity"]),
        _table("efficiency", manifest, ["latency_ms", "throughput", "gpu_memory_mb"], ["dataset", "method"]),
        _table("diagnostic", manifest, ["validity_rate", "hallucination_rate"], ["dataset", "method"]),
    ]
    plan = {
        NO_EXECUTION_FLAG: True,
        "manual_metric_values_allowed": False,
        "numeric_values_present": False,
        "protocol_version": manifest.get("protocol_version"),
        "status": "PLANNED_NO_NUMBERS",
        "tables": tables,
    }
    write_json(output, plan)
    return plan


def _table(name: str, manifest: dict[str, Any], metric_columns: list[str], grouping_keys: list[str]) -> dict[str, Any]:
    base = f"outputs/tables/paper/{manifest.get('protocol_version', 'protocol_v1')}/{name}"
    return {
        "grouping_keys": grouping_keys,
        "input_metrics_files": [
            f"{experiment.get('output_dir')}/**/metrics.json"
            for experiment in manifest.get("experiments", [])
        ],
        "metric_columns": metric_columns,
        "name": name,
        "output_csv": f"{base}.csv",
        "output_tex": f"{base}.tex",
        "significance_markers": "paired bootstrap or paired randomization from metrics files",
        "values": [],
    }
