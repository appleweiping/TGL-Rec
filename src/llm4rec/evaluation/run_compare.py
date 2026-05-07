"""Compare prediction runs on aligned same-candidate events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import ensure_dir, iter_jsonl, write_csv_rows, write_json
from llm4rec.metrics.ranking import deduplicate_preserve_order


@dataclass(frozen=True)
class PredictionRunSpec:
    """One prediction JSONL to compare."""

    name: str
    path: Path


def compare_prediction_runs(
    *,
    runs: list[PredictionRunSpec],
    output_dir: str | Path,
    baseline: str | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
    strict: bool = True,
) -> dict[str, Any]:
    """Align prediction rows and export per-event/per-method comparison tables."""

    if len(runs) < 2:
        raise ValueError("At least two prediction runs are required.")
    run_rows = {run.name: _load_indexed_rows(run) for run in runs}
    key_sets = {name: set(rows) for name, rows in run_rows.items()}
    common = set.intersection(*key_sets.values())
    if strict:
        missing = {
            name: sorted(set.union(*key_sets.values()) - keys)[:10]
            for name, keys in key_sets.items()
            if keys != set.union(*key_sets.values())
        }
        if missing:
            raise ValueError(f"Prediction runs are not aligned; missing key preview: {missing}")
    if not common:
        raise ValueError("Prediction runs have no aligned events.")
    output = ensure_dir(output_dir)
    baseline_name = baseline or runs[0].name
    if baseline_name not in run_rows:
        raise ValueError(f"Unknown baseline run: {baseline_name}")

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for name, rows in sorted(run_rows.items()):
        ranks = []
        hits = {k: 0.0 for k in ks}
        baseline_deltas = {k: 0.0 for k in ks}
        for key in sorted(common):
            row = rows[key]
            target_rank = _target_rank(row)
            baseline_rank = _target_rank(run_rows[baseline_name][key])
            ranks.append(target_rank)
            record = {
                "candidate_count": len(row.get("candidate_items", [])),
                "dataset": key[0],
                "event_id": key[1],
                "method": name,
                "source_event_id": key[2],
                "split": row.get("split", ""),
                "target_item": key[4],
                "target_rank": target_rank if target_rank is not None else "",
                "user_id": key[3],
            }
            for k in ks:
                hit = float(target_rank is not None and target_rank <= k)
                base_hit = float(baseline_rank is not None and baseline_rank <= k)
                hits[k] += hit
                baseline_deltas[k] += hit - base_hit
                record[f"hit@{k}"] = hit
                record[f"delta_hit@{k}_vs_{baseline_name}"] = hit - base_hit
            event_rows.append(record)
        summary = {
            "aligned_events": len(common),
            "baseline": baseline_name,
            "mean_target_rank": _mean([rank for rank in ranks if rank is not None]),
            "method": name,
            "missing_target_rate": sum(1 for rank in ranks if rank is None) / float(len(ranks)),
        }
        for k in ks:
            summary[f"hit@{k}"] = hits[k] / float(len(common))
            summary[f"delta_hit@{k}_vs_{baseline_name}"] = baseline_deltas[k] / float(len(common))
        summary_rows.append(summary)

    write_csv_rows(output / "aligned_event_comparison.csv", event_rows)
    write_csv_rows(output / "method_summary.csv", summary_rows)
    manifest = {
        "baseline": baseline_name,
        "ks": list(ks),
        "num_aligned_events": len(common),
        "runs": [{"name": run.name, "path": str(run.path)} for run in runs],
        "strict": strict,
    }
    write_json(output / "comparison_manifest.json", manifest)
    return {"manifest": manifest, "summary_rows": summary_rows}


def _load_indexed_rows(run: PredictionRunSpec) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(run.path):
        key = (
            str(row.get("dataset") or row.get("domain") or ""),
            str(row.get("event_id") or ""),
            str(row.get("source_event_id") or ""),
            str(row.get("user_id") or ""),
            str(row.get("target_item") or ""),
        )
        if key in rows:
            raise ValueError(f"Duplicate aligned event key in {run.name}: {key}")
        rows[key] = row
    return rows


def _target_rank(row: dict[str, Any]) -> int | None:
    target = str(row.get("target_item", ""))
    for rank, item in enumerate(deduplicate_preserve_order(row.get("predicted_items", [])), start=1):
        if item == target:
            return rank
    return None


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
