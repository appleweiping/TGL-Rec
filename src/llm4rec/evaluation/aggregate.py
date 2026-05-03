"""Aggregate protocol-v1 paper runs across seeds."""

from __future__ import annotations

import csv
import json
import math
import shutil
from array import array
from pathlib import Path
from typing import Any

from llm4rec.evaluation.failure_audit import audit_failures
from llm4rec.evaluation.significance import PairedMoments, paired_t_test_from_moments
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, sha256_file, write_csv_rows, write_json


MULTISEED_METRICS = [
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "HitRate@1",
    "HitRate@5",
    "HitRate@10",
    "NDCG@1",
    "NDCG@5",
    "NDCG@10",
    "MRR@10",
    "coverage",
    "catalog_coverage",
    "novelty",
    "long_tail_ratio",
    "validity_rate",
    "hallucination_rate",
    "runtime_seconds",
]
SIGNIFICANCE_METRICS = ["Recall@5", "NDCG@5", "MRR@10"]
OURS_METHODS = ["time_graph_evidence", "time_graph_evidence_dynamic"]
BASELINE_METHODS = ["popularity", "bm25", "mf_bpr", "sasrec", "temporal_graph_encoder"]


def aggregate_multiseed_results(
    *,
    seed0_dir: str | Path,
    multiseed_dir: str | Path,
    seeds: list[int],
) -> dict[str, Any]:
    """Aggregate seed-level paper matrix outputs into mean/std and tests."""

    root = ensure_dir(resolve_path(multiseed_dir))
    seed0_source = resolve_path(seed0_dir)
    seed_dirs = {int(seed): _seed_dir(root, int(seed), seed0_source) for seed in seeds}
    if 0 in seed_dirs:
        _write_seed0_reference(root / "seed_0", seed0_source)

    status_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = seed_dirs[int(seed)]
        status_rows.extend(_status_rows_for_seed(seed_dir, int(seed)))
        metric_rows.extend(_metric_rows_for_seed(seed_dir, int(seed)))

    write_csv_rows(root / "method_status.csv", status_rows, fieldnames=_status_fieldnames())
    write_csv_rows(root / "metrics_by_seed.csv", metric_rows)

    aggregate_rows = aggregate_metric_rows(metric_rows, seeds=seeds, metric_names=MULTISEED_METRICS)
    write_csv_rows(root / "aggregate_metrics.csv", aggregate_rows, fieldnames=_aggregate_fieldnames())
    write_json(
        root / "aggregate_metrics.json",
        {
            "label": "PAPER-SCALE MULTI-SEED MAIN ACCURACY RESULTS, PROTOCOL_V1",
            "metrics": aggregate_rows,
            "seeds": [int(seed) for seed in seeds],
        },
    )

    integrity = collect_artifact_integrity(seed_dirs)
    write_json(root / "artifact_integrity.json", integrity)

    significance_rows, significance_json = compute_significance_tests(
        status_rows=status_rows,
        aggregate_rows=aggregate_rows,
        seeds=[int(seed) for seed in seeds],
    )
    write_csv_rows(root / "significance_tests.csv", significance_rows, fieldnames=_significance_fieldnames())
    write_json(root / "significance_tests.json", significance_json)

    _write_multiseed_manifest(root, seed0_source, seed_dirs, seeds, status_rows, integrity)
    failure_report = audit_failures(root)
    return {
        "aggregate_metrics": str(root / "aggregate_metrics.csv"),
        "artifact_integrity": integrity,
        "failure_report": failure_report,
        "run_dir": str(root),
        "significance_tests": str(root / "significance_tests.csv"),
    }


def aggregate_metric_rows(
    metric_rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    """Aggregate metric rows by dataset/method/metric."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    rows: list[dict[str, Any]] = []
    expected_seeds = {int(seed) for seed in seeds}
    for (dataset, method), group in sorted(grouped.items()):
        present_seeds = {int(row.get("seed", -1)) for row in group}
        missing = sorted(expected_seeds - present_seeds)
        for metric in metric_names:
            values = [
                float(row.get(metric, 0.0) or 0.0)
                for row in sorted(group, key=lambda value: int(value.get("seed", 0)))
                if str(row.get("status", "succeeded")).lower() == "succeeded"
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "mean": _mean(values),
                    "missing_seeds": " ".join(str(seed) for seed in missing),
                    "num_seeds": len(values),
                    "seeds": " ".join(str(row.get("seed")) for row in sorted(group, key=lambda value: int(value.get("seed", 0)))),
                    "std": _sample_std(values),
                }
            )
    return rows


def collect_artifact_integrity(seed_dirs: dict[int, Path]) -> dict[str, Any]:
    """Collect pre/post checksum reports for every included seed."""

    seed_reports: dict[str, Any] = {}
    all_unchanged = True
    all_verified = True
    for seed, seed_dir in sorted(seed_dirs.items()):
        pre_path = seed_dir / "artifact_checksums_pre.json"
        post_path = seed_dir / "artifact_checksums_post.json"
        pre = _read_json(pre_path)
        post = _read_json(post_path)
        method_checksums = _artifact_checksums_from_methods(seed_dir)
        if method_checksums:
            pre = {**method_checksums, **dict(pre or {})}
            post = {**method_checksums, **dict(post or {})}
        unchanged = bool(pre and post and pre == post)
        verified = bool(pre and _has_candidate_checksums(pre))
        all_unchanged = all_unchanged and unchanged
        all_verified = all_verified and verified
        seed_reports[str(seed)] = {
            "artifact_checksums_post": post,
            "artifact_checksums_pre": pre,
            "candidate_checksums_verified": verified,
            "seed_dir": str(seed_dir),
            "unchanged": unchanged,
        }
    return {
        "all_artifacts_unchanged": all_unchanged,
        "candidate_checksums_verified": all_verified,
        "protocol_version": "protocol_v1",
        "seeds": seed_reports,
    }


def compute_significance_tests(
    *,
    status_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run paired tests over aligned per-user prediction contributions."""

    status_index = {
        (
            str(row.get("dataset", "")),
            int(row.get("seed", -1)),
            str(row.get("method", "")),
        ): row
        for row in status_rows
        if str(row.get("status", "")).lower() == "succeeded"
    }
    datasets = sorted({key[0] for key in status_index})
    aggregate_index = {
        (str(row.get("dataset", "")), str(row.get("method", "")), str(row.get("metric", ""))): row
        for row in aggregate_rows
    }
    csv_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        needed_methods = sorted(
            {
                *OURS_METHODS,
                *BASELINE_METHODS,
                *[
                    str(_best_baseline(dataset, metric, aggregate_index) or "")
                    for metric in SIGNIFICANCE_METRICS
                ],
            }
            - {""}
        )
        contributions, warnings = _load_dataset_contributions(
            dataset=dataset,
            seeds=seeds,
            methods=needed_methods,
            status_index=status_index,
        )
        for method_a in OURS_METHODS:
            for metric in SIGNIFICANCE_METRICS:
                best = _best_baseline(dataset, metric, aggregate_index)
                comparators = _dedupe([best, "sasrec", "bm25", "temporal_graph_encoder"])
                for method_b in comparators:
                    row, payload = _paired_result_row(
                        contributions=contributions,
                        dataset=dataset,
                        metric=metric,
                        method_a=method_a,
                        method_b=method_b,
                        seeds=seeds,
                        notes_prefix="best_non_ours" if method_b == best else "planned_comparator",
                    )
                    if warnings:
                        row["notes"] = _join_notes(row.get("notes", ""), "; ".join(warnings))
                        payload["warnings"] = warnings
                    csv_rows.append(row)
                    json_rows.append(payload)
    return csv_rows, {"comparisons": json_rows, "metrics": SIGNIFICANCE_METRICS}


def _paired_result_row(
    *,
    contributions: dict[tuple[int, str, str], array],
    dataset: str,
    metric: str,
    method_a: str,
    method_b: str,
    seeds: list[int],
    notes_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moments = PairedMoments()
    used_seeds: list[int] = []
    missing: list[str] = []
    for seed in seeds:
        ours = contributions.get((int(seed), method_a, metric))
        baseline = contributions.get((int(seed), method_b, metric))
        if ours is None or baseline is None:
            missing.append(str(seed))
            continue
        if len(ours) != len(baseline):
            missing.append(f"{seed}:length_mismatch")
            continue
        for baseline_value, ours_value in zip(baseline, ours):
            moments.add(baseline_value, ours_value)
        used_seeds.append(int(seed))
    result = paired_t_test_from_moments(moments)
    notes = notes_prefix
    if missing:
        notes = _join_notes(notes, f"missing_or_mismatched_seeds={' '.join(missing)}")
    notes = _join_notes(notes, f"n={result.get('n', 0)}")
    if result.get("notes"):
        notes = _join_notes(notes, str(result["notes"]))
    row = {
        "dataset": dataset,
        "effect_direction": result.get("effect_direction", "insufficient_sample"),
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "notes": notes,
        "p_value": result.get("p_value"),
        "significant_at_0_05": result.get("significant_at_0_05", False),
        "test_name": result.get("test", "paired_t_test"),
    }
    payload = {**row, "mean_delta": result.get("mean_delta"), "n": result.get("n"), "seeds": used_seeds}
    return row, payload


def _load_dataset_contributions(
    *,
    dataset: str,
    seeds: list[int],
    methods: list[str],
    status_index: dict[tuple[str, int, str], dict[str, Any]],
) -> tuple[dict[tuple[int, str, str], array], list[str]]:
    contributions: dict[tuple[int, str, str], array] = {}
    warnings: list[str] = []
    for seed in seeds:
        reference_keys: list[str] | None = None
        reference_method: str | None = None
        for method in methods:
            status = status_index.get((dataset, int(seed), method))
            if not status:
                warnings.append(f"missing_status dataset={dataset} seed={seed} method={method}")
                continue
            predictions_path = resolve_path(str(status.get("predictions_path", "")))
            if not predictions_path.is_file():
                warnings.append(f"missing_predictions dataset={dataset} seed={seed} method={method}")
                continue
            method_values = {metric: array("f") for metric in SIGNIFICANCE_METRICS}
            count = 0
            with predictions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = f"{row.get('user_id', '')}\t{row.get('target_item', '')}"
                    if reference_keys is None:
                        reference_keys = []
                        reference_method = method
                    if method == reference_method:
                        reference_keys.append(key)
                    elif count >= len(reference_keys) or reference_keys[count] != key:
                        raise ValueError(
                            "prediction order mismatch for paired significance: "
                            f"dataset={dataset} seed={seed} method={method} row={count}"
                        )
                    for metric in SIGNIFICANCE_METRICS:
                        method_values[metric].append(_prediction_contribution(row, metric))
                    count += 1
            if method != reference_method and reference_keys is not None and count != len(reference_keys):
                raise ValueError(
                    "prediction length mismatch for paired significance: "
                    f"dataset={dataset} seed={seed} method={method} count={count} expected={len(reference_keys)}"
                )
            for metric, values in method_values.items():
                contributions[(int(seed), method, metric)] = values
    return contributions, warnings


def _prediction_contribution(row: dict[str, Any], metric: str) -> float:
    target = str(row.get("target_item", ""))
    predicted = [str(item) for item in row.get("predicted_items", [])]
    rank = None
    for index, item in enumerate(predicted[:10], start=1):
        if item == target:
            rank = index
            break
    if metric == "Recall@5":
        return 1.0 if rank is not None and rank <= 5 else 0.0
    if metric == "NDCG@5":
        return 1.0 / math.log2(rank + 1.0) if rank is not None and rank <= 5 else 0.0
    if metric == "MRR@10":
        return 1.0 / float(rank) if rank is not None and rank <= 10 else 0.0
    raise ValueError(f"unsupported significance metric: {metric}")


def _write_seed0_reference(seed0_dir: Path, source_dir: Path) -> None:
    ensure_dir(seed0_dir)
    for name in ("method_status.csv", "metrics_by_method.csv", "artifact_checksums_pre.json", "artifact_checksums_post.json"):
        source = source_dir / name
        if source.is_file():
            shutil.copyfile(source, seed0_dir / name)
    payload = {
        "mode": "referenced_existing_outputs",
        "source_path": str(source_dir),
        "source_sha256": {
            name: sha256_file(source_dir / name)
            for name in ("method_status.csv", "metrics_by_method.csv")
            if (source_dir / name).is_file()
        },
    }
    write_json(seed0_dir / "linked_or_copied_from_main_accuracy_seed0.json", payload)
    (seed0_dir / "source_path.txt").write_text(str(source_dir) + "\n", encoding="utf-8", newline="\n")


def _seed_dir(root: Path, seed: int, seed0_source: Path) -> Path:
    return seed0_source if int(seed) == 0 else root / f"seed_{int(seed)}"


def _status_rows_for_seed(seed_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows = _read_csv(seed_dir / "method_status.csv")
    output = []
    for row in rows:
        output.append(
            {
                "checkpoint_path": row.get("checkpoint_path", ""),
                "dataset": row.get("dataset", ""),
                "failure_reason": row.get("failure_reason", ""),
                "message": row.get("message", ""),
                "method": row.get("method", ""),
                "metrics_path": row.get("metrics_path", ""),
                "predictions_path": row.get("predictions_path", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "seed": seed,
                "seed_dir": str(seed_dir),
                "status": row.get("status", "failed"),
            }
        )
    return output


def _metric_rows_for_seed(seed_dir: Path, seed: int) -> list[dict[str, Any]]:
    status_by_key = {
        (row.get("dataset", ""), row.get("method", "")): row.get("status", "")
        for row in _read_csv(seed_dir / "method_status.csv")
    }
    rows = _read_csv(seed_dir / "metrics_by_method.csv")
    output = []
    for row in rows:
        key = (row.get("dataset", ""), row.get("method", ""))
        row = dict(row)
        row["seed"] = seed
        row["status"] = status_by_key.get(key, "succeeded")
        output.append(row)
    return output


def _best_baseline(
    dataset: str,
    metric: str,
    aggregate_index: dict[tuple[str, str, str], dict[str, Any]],
) -> str | None:
    best_method = None
    best_value = -math.inf
    for method in BASELINE_METHODS:
        row = aggregate_index.get((dataset, method, metric))
        if not row:
            continue
        value = float(row.get("mean", 0.0) or 0.0)
        if value > best_value:
            best_value = value
            best_method = method
    return best_method


def _write_multiseed_manifest(
    root: Path,
    seed0_source: Path,
    seed_dirs: dict[int, Path],
    seeds: list[int],
    status_rows: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> None:
    write_json(
        root / "run_manifest.json",
        {
            "api_calls_allowed": False,
            "artifact_integrity": integrity,
            "datasets": sorted({str(row.get("dataset", "")) for row in status_rows}),
            "llm_provider": "none",
            "lora_training_enabled": False,
            "matrix": "main_accuracy",
            "methods": sorted({str(row.get("method", "")) for row in status_rows}),
            "protocol_version": "protocol_v1",
            "seed0_source": str(seed0_source),
            "seed_dirs": {str(seed): str(path) for seed, path in sorted(seed_dirs.items())},
            "seeds": [int(seed) for seed in seeds],
        },
    )


def _has_candidate_checksums(payload: dict[str, Any]) -> bool:
    for value in payload.values():
        if not isinstance(value, dict):
            return False
        if not value.get("candidate_sha256"):
            return False
    return True


def _artifact_checksums_from_methods(seed_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path in seed_dir.glob("*/*/artifact_checksums.json"):
        payload = _read_json(path)
        dataset = str(payload.get("dataset") or path.parent.parent.name)
        if not payload.get("candidate_sha256"):
            continue
        output.setdefault(
            dataset,
            {
                key: value
                for key, value in payload.items()
                if key
                in {
                    "candidate_artifact",
                    "candidate_pool_artifact",
                    "candidate_pool_sha256",
                    "candidate_sha256",
                    "dataset",
                    "split_artifact",
                    "split_sha256",
                }
            },
        )
    return output


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values) or 1)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / float(len(values) - 1)) ** 0.5


def _dedupe(values: list[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(str(value))
    return output


def _join_notes(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left}; {right}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _status_fieldnames() -> list[str]:
    return [
        "seed",
        "dataset",
        "method",
        "status",
        "runtime_seconds",
        "metrics_path",
        "predictions_path",
        "checkpoint_path",
        "failure_reason",
        "message",
        "seed_dir",
    ]


def _aggregate_fieldnames() -> list[str]:
    return ["dataset", "method", "metric", "mean", "std", "num_seeds", "seeds", "missing_seeds"]


def _significance_fieldnames() -> list[str]:
    return [
        "dataset",
        "metric",
        "method_a",
        "method_b",
        "test_name",
        "p_value",
        "significant_at_0_05",
        "effect_direction",
        "notes",
    ]
