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
    allow_non_reportable: bool = False,
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
    _validate_fairness_invariants(run_rows, common, allow_non_reportable=allow_non_reportable)
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
        "allow_non_reportable": allow_non_reportable,
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


def _validate_fairness_invariants(
    run_rows: dict[str, dict[tuple[str, str, str, str, str], dict[str, Any]]],
    common: set[tuple[str, str, str, str, str]],
    *,
    allow_non_reportable: bool,
) -> None:
    for key in common:
        reference_row: dict[str, Any] | None = None
        for name, rows in sorted(run_rows.items()):
            row = rows[key]
            if reference_row is None:
                reference_row = row
            else:
                _assert_same_value(name, key, "protocol_version", reference_row, row)
                _assert_same_value(name, key, "split", reference_row, row)
                if [str(item) for item in row.get("candidate_items", [])] != [
                    str(item) for item in reference_row.get("candidate_items", [])
                ]:
                    raise ValueError(f"Candidate set mismatch for run={name} key={key}")
            if not allow_non_reportable and _row_is_non_reportable(row):
                raise ValueError(
                    f"Refusing to compare non-reportable/scaffold row for run={name} key={key}. "
                    "Pass allow_non_reportable=True only for diagnostic comparisons."
                )


def _assert_same_value(
    run_name: str,
    key: tuple[str, str, str, str, str],
    field: str,
    reference_row: dict[str, Any],
    row: dict[str, Any],
) -> None:
    left = str(reference_row.get(field, ""))
    right = str(row.get(field, ""))
    if left != right:
        raise ValueError(f"{field} mismatch for run={run_name} key={key}: {left!r} != {right!r}")


def _row_is_non_reportable(row: dict[str, Any]) -> bool:
    metadata = dict(row.get("metadata", {}))
    provenance = dict(metadata.get("baseline_provenance", {}))
    if bool(provenance.get("do_not_merge_into_main_accuracy_table", False)):
        return True
    if bool(provenance.get("scaffold_only", False)):
        return True
    if provenance and provenance.get("reportable_baseline") is False:
        return True
    return bool(metadata.get("do_not_merge_into_main_accuracy_table", False))


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
