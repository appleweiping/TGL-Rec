"""Shared evaluator for Phase 1 prediction JSONL files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from llm4rec.evaluation.candidate_resolver import CandidateResolver
from llm4rec.evaluation.export import write_evaluation_outputs
from llm4rec.evaluation.prediction_schema import PredictionSchemaError, validate_prediction_row
from llm4rec.io.artifacts import read_jsonl
from llm4rec.metrics.ranking import aggregate_ranking_metrics, coverage


def evaluate_predictions(
    *,
    predictions_path: str | Path,
    item_catalog_path: str | Path,
    output_dir: str | Path,
    ks: tuple[int, ...],
    candidate_protocol: str,
) -> dict[str, Any]:
    """Validate predictions, compute metrics, and write evaluation outputs."""

    raw_rows = read_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        try:
            rows.append(validate_prediction_row(row, candidate_protocol=candidate_protocol))
        except PredictionSchemaError as exc:
            raise PredictionSchemaError(f"invalid prediction row {index}: {exc}") from exc
    if not rows:
        raise ValueError(f"No prediction rows found: {predictions_path}")

    item_rows = read_jsonl(item_catalog_path)
    item_catalog = {str(row["item_id"]) for row in item_rows}
    resolver_cache: dict[tuple[str, str], CandidateResolver] = {}
    overall = _metrics_for_rows(
        rows,
        item_catalog=item_catalog,
        ks=ks,
        candidate_protocol=candidate_protocol,
        resolver_cache=resolver_cache,
    )
    by_domain_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain_rows[str(row.get("domain") or "unknown")].append(row)
    by_domain = {
        domain: _metrics_for_rows(
            domain_rows,
            item_catalog=item_catalog,
            ks=ks,
            candidate_protocol=candidate_protocol,
            resolver_cache=resolver_cache,
        )
        for domain, domain_rows in sorted(by_domain_rows.items())
    }
    by_method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_rows[str(row.get("method") or "unknown")].append(row)
    by_method = {
        method: _metrics_for_rows(
            method_rows,
            item_catalog=item_catalog,
            ks=ks,
            candidate_protocol=candidate_protocol,
            resolver_cache=resolver_cache,
        )
        for method, method_rows in sorted(by_method_rows.items())
    }
    candidate_metadata = _candidate_metadata(rows, resolver_cache)
    metrics = {
        "by_domain": by_domain,
        "by_method": by_method,
        **candidate_metadata,
        "candidate_protocol": candidate_protocol,
        "num_predictions": len(rows),
        "overall": overall,
    }
    write_evaluation_outputs(output_dir, metrics)
    return metrics


def _metrics_for_rows(
    rows: list[dict[str, Any]],
    *,
    item_catalog: set[str],
    ks: tuple[int, ...],
    candidate_protocol: str,
    resolver_cache: dict[tuple[str, str], CandidateResolver],
) -> dict[str, float]:
    metrics = aggregate_ranking_metrics(rows, ks=ks)
    metrics.update(_validity_metrics(rows, item_catalog=item_catalog, candidate_protocol=candidate_protocol, resolver_cache=resolver_cache))
    metrics["coverage"] = coverage(rows, item_catalog)
    return dict(sorted(metrics.items()))


def _validity_metrics(
    rows: list[dict[str, Any]],
    *,
    item_catalog: set[str],
    candidate_protocol: str,
    resolver_cache: dict[tuple[str, str], CandidateResolver],
) -> dict[str, float]:
    total = 0
    invalid = 0
    candidate_violations = 0
    for row in rows:
        candidates = _candidate_items(row, resolver_cache)
        candidate_set = {str(item) for item in candidates}
        if candidate_protocol != "no_candidates" and row["target_item"] not in candidate_set:
            raise PredictionSchemaError(f"target_item missing from resolved candidates: {row['target_item']}")
        for item_id in row.get("predicted_items", []):
            item = str(item_id)
            total += 1
            in_catalog = item in item_catalog
            in_candidates = candidate_protocol == "no_candidates" or not candidates or item in candidate_set
            if not in_candidates:
                candidate_violations += 1
            if not in_catalog or not in_candidates:
                invalid += 1
    if total == 0:
        return {"candidate_adherence_rate": 1.0, "hallucination_rate": 0.0, "validity_rate": 1.0}
    return {
        "candidate_adherence_rate": 1.0 - candidate_violations / float(total),
        "hallucination_rate": invalid / float(total),
        "validity_rate": 1.0 - invalid / float(total),
    }


def _candidate_items(
    row: dict[str, Any],
    resolver_cache: dict[tuple[str, str], CandidateResolver],
) -> list[str]:
    if row.get("candidate_items"):
        return [str(item) for item in row["candidate_items"]]
    candidate_ref = row.get("candidate_ref")
    if not isinstance(candidate_ref, dict):
        return []
    key = (str(candidate_ref["artifact_path"]), str(candidate_ref["artifact_sha256"]).lower())
    resolver = resolver_cache.get(key)
    if resolver is None:
        resolver = CandidateResolver.from_ref(candidate_ref)
        resolver_cache[key] = resolver
    return resolver.resolve_prediction_row(row)


def _candidate_metadata(
    rows: list[dict[str, Any]],
    resolver_cache: dict[tuple[str, str], CandidateResolver],
) -> dict[str, Any]:
    compact_refs = [row.get("candidate_ref") for row in rows if isinstance(row.get("candidate_ref"), dict)]
    if not compact_refs:
        return {"candidate_schema": "expanded", "resolver_mode": "expanded"}
    first = compact_refs[0]
    key = (str(first["artifact_path"]), str(first["artifact_sha256"]).lower())
    resolver = resolver_cache.get(key)
    resolver_mode = resolver.resolver_mode if resolver is not None else "compact_ref"
    return {
        "candidate_artifact_path": str(first["artifact_path"]),
        "candidate_artifact_sha256": str(first["artifact_sha256"]).lower(),
        "candidate_schema": "compact_ref_v1" if len(compact_refs) == len(rows) else "mixed",
        "resolver_mode": resolver_mode,
    }
