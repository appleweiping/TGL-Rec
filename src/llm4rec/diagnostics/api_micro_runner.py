"""Phase 3B OpenAI-compatible API micro-diagnostic runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from llm4rec.diagnostics.api_case_sampler import (
    load_interaction_catalog,
    load_item_records,
    load_time_window_edges,
    load_transition_edges,
    parse_time_bucket_pairs,
    prompt_example_from_api_sample,
    sample_api_micro_cases,
)
from llm4rec.diagnostics.api_result_audit import (
    audit_api_micro_response,
    raw_output_row,
)
from llm4rec.diagnostics.llm_grounding import build_edge_index
from llm4rec.diagnostics.llm_sequence_time_runner import _load_llm_config, _prepare_source_run
from llm4rec.evaluation.api_micro_evaluator import evaluate_api_micro_predictions
from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.llm.api_guard import (
    build_api_guard_config,
    validate_api_guard,
    validate_dry_run_config,
)
from llm4rec.llm.base import LLMRequest
from llm4rec.llm.cost_estimator import (
    CostPreflight,
    assert_within_call_cap,
    build_cost_preflight,
)
from llm4rec.llm.cost_tracker import CostLatencyTracker
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.llm.response_cache import ResponseCache
from llm4rec.llm.structured_output import structured_output_metadata
from llm4rec.prompts.builder import build_prompt
from llm4rec.prompts.variants import get_prompt_variant
from llm4rec.utils.env import collect_environment


DEFAULT_API_PROMPT_VARIANTS = [
    "history_only",
    "history_with_order",
    "history_with_time_buckets",
    "history_with_transition_evidence",
    "history_with_contrastive_evidence",
]

EMPTY_METRIC_FIELDS = [
    "case_group",
    "prompt_variant",
    "num_predictions",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "NDCG@1",
    "NDCG@3",
    "NDCG@5",
    "MRR@1",
    "MRR@3",
    "MRR@5",
    "validity_rate",
    "hallucination_rate",
    "parse_success_rate",
    "candidate_adherence_rate",
    "evidence_grounding_rate",
    "transition_evidence_usage_rate",
    "time_evidence_usage_rate",
    "semantic_evidence_usage_rate",
    "contrastive_evidence_usage_rate",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_rate",
]


def estimate_api_micro_cost(config_path: str | Path) -> tuple[Path, CostPreflight]:
    """Sample cases, build prompts, and save only the cost preflight."""

    prepared = _prepare_api_micro_run(config_path)
    _write_preflight(prepared["run_dir"], prepared["preflight"])
    _write_sampled_cases(prepared["run_dir"], prepared["samples"])
    print_preflight(prepared["preflight"])
    assert_within_call_cap(prepared["preflight"])
    return prepared["run_dir"], prepared["preflight"]


def run_api_micro_diagnostic(config_path: str | Path, *, dry_run: bool = False) -> Path:
    """Run Phase 3B dry-run or real API micro diagnostics."""

    prepared = _prepare_api_micro_run(config_path)
    config = prepared["config"]
    llm_config = prepared["llm_config"]
    run_dir = prepared["run_dir"]
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")
    logger = RunLogger(run_dir / "logs.txt")
    logger.info(f"starting Phase 3B API micro diagnostic dry_run={dry_run}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    _write_sampled_cases(run_dir, prepared["samples"])
    _write_preflight(run_dir, prepared["preflight"])
    print_preflight(prepared["preflight"])
    assert_within_call_cap(prepared["preflight"])

    guard = build_api_guard_config(
        config=config,
        llm_config=llm_config,
        preflight=prepared["preflight"],
        run_dir=run_dir,
    )
    guard_warnings = validate_dry_run_config(guard)
    if dry_run:
        _write_planned_request_files(run_dir, prepared["request_rows"])
        _write_empty_execution_outputs(run_dir, diagnostics_dir, guard_warnings)
        logger.info("Phase 3B dry-run completed without API calls")
        return run_dir

    validate_api_guard(guard)
    _execute_api_requests(
        config=config,
        llm_config=llm_config,
        prepared=prepared,
        logger=logger,
    )
    logger.info("Phase 3B API micro diagnostic completed")
    return run_dir


def print_preflight(preflight: CostPreflight) -> None:
    """Print the preflight in a compact, script-friendly format."""

    data = preflight.to_dict()
    print("Phase 3B API micro cost preflight")
    for key in (
        "number_of_cases",
        "prompt_variants",
        "estimated_api_calls",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
        "max_api_calls",
        "cache_enabled",
        "cache_policy",
        "model_name",
        "run_dir",
    ):
        print(f"{key}: {data[key]}")


def _prepare_api_micro_run(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    experiment = dict(config.get("experiment", {}))
    seed = int(experiment.get("seed", 2026))
    set_global_seed(seed)
    run_id = str(experiment.get("run_id", "phase3b_api_micro"))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = ensure_dir(output_root / run_id)
    logger = RunLogger(run_dir / "logs.txt")
    source_run_dir = _prepare_source_run(config, logger)
    samples = sample_api_micro_cases(source_run_dir=source_run_dir, config=config)
    if not samples:
        raise ValueError("No Phase 3B API micro samples could be built from source artifacts.")
    llm_config = _load_llm_config(config)
    prompt_variants = _prompt_variants(config)
    item_records = load_item_records(source_run_dir)
    prompt_requests: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    prompts: list[str] = []
    cache = _make_cache(config)
    for sample in samples:
        example = prompt_example_from_api_sample(sample, item_records)
        for prompt_variant in prompt_variants:
            prompt = build_prompt(
                example,
                prompt_variant=prompt_variant,
                max_history_items=int(config.get("diagnostic", {}).get("max_history_items", 20)),
                max_evidence_items=int(config.get("diagnostic", {}).get("max_evidence_items", 8)),
            )
            request = _make_request(
                prompt=prompt,
                config=config,
                llm_config=llm_config,
                sample=sample,
                prompt_variant=prompt_variant,
                run_dir=run_dir,
            )
            prompt_requests.append({"prompt": prompt, "request": request, "sample": sample})
            request_rows.append(_request_audit_row(request, sample=sample, cache=cache))
            prompts.append(prompt.prompt)
    controls = dict(config.get("api_safety", {}))
    decoding = dict(llm_config.get("decoding", {}))
    preflight = build_cost_preflight(
        prompts=prompts,
        number_of_cases=len(samples),
        prompt_variants=prompt_variants,
        max_tokens=int(decoding.get("max_tokens", 512)),
        max_api_calls=int(controls.get("max_api_calls", 125)),
        cache_enabled=bool(config.get("cache", {}).get("enabled", True)),
        cache_policy=str(config.get("cache", {}).get("policy", "missing")),
        model_name=str(llm_config.get("model")),
        run_dir=str(run_dir),
    )
    return {
        "config": config,
        "item_catalog": load_interaction_catalog(source_run_dir),
        "llm_config": llm_config,
        "preflight": preflight,
        "prompt_requests": prompt_requests,
        "request_rows": request_rows,
        "run_dir": run_dir,
        "samples": samples,
        "source_run_dir": source_run_dir,
    }


def _execute_api_requests(
    *,
    config: dict[str, Any],
    llm_config: dict[str, Any],
    prepared: dict[str, Any],
    logger: RunLogger,
) -> None:
    run_dir: Path = prepared["run_dir"]
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")
    source_run_dir = Path(prepared["source_run_dir"])
    cache = _make_cache(config)
    provider = _make_provider(llm_config, config=config, cache=cache)
    tracker = CostLatencyTracker(pricing=dict(llm_config.get("pricing", {})))
    transition_index = build_edge_index(load_transition_edges(source_run_dir))
    time_window_index = build_edge_index(load_time_window_edges(source_run_dir))
    on_api_error = str(config.get("api_safety", {}).get("on_api_error", "continue"))
    max_api_calls = int(config.get("api_safety", {}).get("max_api_calls", 125))
    completed = (
        _load_completed_predictions(run_dir)
        if bool(config.get("experiment", {}).get("resume", False))
        else []
    )
    completed_keys = {_prediction_key(row) for row in completed}
    predictions = list(completed)
    raw_output_rows: list[dict[str, Any]] = []
    parse_failure_rows: list[dict[str, Any]] = []
    hallucination_rows: list[dict[str, Any]] = []
    api_failure_rows: list[dict[str, Any]] = []
    api_call_count = 0

    write_jsonl(run_dir / "api_requests.jsonl", prepared["request_rows"])
    for planned in prepared["prompt_requests"]:
        sample = planned["sample"]
        request: LLMRequest = planned["request"]
        prompt_variant = str(request.metadata["prompt_variant"])
        key = (str(sample["sample_id"]), prompt_variant)
        if key in completed_keys:
            continue
        cached = cache.get(request)
        if cached is None and api_call_count >= max_api_calls:
            raise RuntimeError(f"max_api_calls hard cap exceeded before request {key}")
        try:
            response = cached if cached is not None else provider.generate(request)
            if cached is None:
                api_call_count += 1
            tracker.record(response)
            raw_output_rows.append(
                raw_output_row(
                    sample_id=str(sample["sample_id"]),
                    prompt_variant=prompt_variant,
                    response=response,
                )
            )
            prediction, parse_failure, hallucination = audit_api_micro_response(
                sample=sample,
                prompt_variant=prompt_variant,
                prompt_version=str(request.prompt_version),
                response=response,
                transition_edges=transition_index,
                time_window_edges=time_window_index,
                time_bucket_by_pair=parse_time_bucket_pairs(sample.get("time_bucket_by_pair", {})),
                run_mode=str(config.get("experiment", {}).get("run_mode", "diagnostic_api")),
            )
            predictions.append(prediction)
            if parse_failure is not None:
                parse_failure_rows.append(parse_failure)
            if hallucination is not None:
                hallucination_rows.append(hallucination)
        except Exception as exc:
            failure = {
                "error": str(exc),
                "on_api_error": on_api_error,
                "prompt_variant": prompt_variant,
                "sample_id": sample["sample_id"],
            }
            api_failure_rows.append(failure)
            logger.info(f"API failure sample={sample['sample_id']} prompt={prompt_variant}: {exc}")
            if on_api_error == "stop":
                break

    _write_real_outputs(
        run_dir=run_dir,
        diagnostics_dir=diagnostics_dir,
        predictions=predictions,
        raw_output_rows=raw_output_rows,
        hallucination_rows=hallucination_rows,
        parse_failure_rows=parse_failure_rows,
        api_failure_rows=api_failure_rows,
        tracker_summary={**tracker.summary(), "api_call_count": api_call_count},
        item_catalog=prepared["item_catalog"],
        config=config,
    )


def _write_real_outputs(
    *,
    run_dir: Path,
    diagnostics_dir: Path,
    predictions: list[dict[str, Any]],
    raw_output_rows: list[dict[str, Any]],
    hallucination_rows: list[dict[str, Any]],
    parse_failure_rows: list[dict[str, Any]],
    api_failure_rows: list[dict[str, Any]],
    tracker_summary: dict[str, Any],
    item_catalog: set[str],
    config: dict[str, Any],
) -> None:
    write_jsonl(run_dir / "api_raw_outputs.jsonl", raw_output_rows)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(diagnostics_dir / "hallucination_cases.jsonl", hallucination_rows)
    write_jsonl(diagnostics_dir / "parse_failures.jsonl", parse_failure_rows)
    write_jsonl(diagnostics_dir / "api_failures.jsonl", api_failure_rows)
    evaluation = dict(config.get("evaluation", {}))
    ks = tuple(int(k) for k in evaluation.get("ks", [1, 3, 5]))
    metric_rows, delta_rows, overlap_rows, comparison_rows = evaluate_api_micro_predictions(
        prediction_rows=predictions,
        item_catalog=item_catalog,
        ks=ks,
        candidate_protocol=str(evaluation.get("candidate_protocol", "full_catalog")),
        top_k=int(evaluation.get("top_k", 5)),
    )
    grounding_rows = _grounding_rows(metric_rows)
    write_json(run_dir / "metrics.json", {"metrics": metric_rows})
    write_csv_rows(run_dir / "metrics.csv", metric_rows, fieldnames=_metric_fieldnames(metric_rows))
    write_csv_rows(
        diagnostics_dir / "prompt_variant_results.csv",
        metric_rows,
        fieldnames=_metric_fieldnames(metric_rows),
    )
    write_csv_rows(diagnostics_dir / "prompt_variant_deltas.csv", delta_rows)
    write_csv_rows(diagnostics_dir / "prediction_overlap_by_prompt.csv", overlap_rows)
    write_csv_rows(diagnostics_dir / "grounding_summary.csv", grounding_rows)
    write_csv_rows(diagnostics_dir / "case_level_comparison.csv", comparison_rows)
    write_json(run_dir / "cost_latency.json", tracker_summary)
    write_json(
        run_dir / "api_micro_summary.json",
        {
            "api_failure_count": len(api_failure_rows),
            "dry_run": False,
            "hallucination_case_count": len(hallucination_rows),
            "metric_rows": len(metric_rows),
            "parse_failure_count": len(parse_failure_rows),
            "prediction_count": len(predictions),
            "tracker": tracker_summary,
        },
    )


def _write_empty_execution_outputs(
    run_dir: Path,
    diagnostics_dir: Path,
    guard_warnings: list[str],
) -> None:
    write_jsonl(run_dir / "api_raw_outputs.jsonl", [])
    write_jsonl(run_dir / "predictions.jsonl", [])
    write_json(run_dir / "metrics.json", {"metrics": []})
    write_csv_rows(run_dir / "metrics.csv", [], fieldnames=EMPTY_METRIC_FIELDS)
    write_json(
        run_dir / "cost_latency.json",
        {
            "cache_hit_count": 0,
            "completion_tokens": 0,
            "estimated_cost": None,
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "prompt_tokens": 0,
            "request_count": 0,
            "total_tokens": 0,
        },
    )
    write_json(
        run_dir / "api_micro_summary.json",
        {
            "dry_run": True,
            "guard_warnings": guard_warnings,
            "message": "Dry run built samples and prompts without API calls.",
        },
    )
    write_csv_rows(
        diagnostics_dir / "prompt_variant_results.csv", [], fieldnames=EMPTY_METRIC_FIELDS
    )
    write_csv_rows(diagnostics_dir / "prompt_variant_deltas.csv", [])
    write_csv_rows(diagnostics_dir / "prediction_overlap_by_prompt.csv", [])
    write_csv_rows(diagnostics_dir / "grounding_summary.csv", [])
    write_jsonl(diagnostics_dir / "hallucination_cases.jsonl", [])
    write_jsonl(diagnostics_dir / "parse_failures.jsonl", [])
    write_jsonl(diagnostics_dir / "api_failures.jsonl", [])
    write_csv_rows(diagnostics_dir / "case_level_comparison.csv", [])


def _write_planned_request_files(run_dir: Path, request_rows: list[dict[str, Any]]) -> None:
    write_jsonl(run_dir / "api_requests.jsonl", request_rows)


def _write_preflight(run_dir: Path, preflight: CostPreflight) -> None:
    write_json(run_dir / "cost_preflight.json", preflight.to_dict())


def _write_sampled_cases(run_dir: Path, samples: list[dict[str, Any]]) -> None:
    write_json(
        run_dir / "artifacts" / "api_sampled_cases.json",
        {"sample_count": len(samples), "samples": samples},
    )


def _prompt_variants(config: dict[str, Any]) -> list[str]:
    diagnostic = dict(config.get("diagnostic", {}))
    return [str(name) for name in diagnostic.get("prompt_variants", DEFAULT_API_PROMPT_VARIANTS)]


def _make_cache(config: dict[str, Any]) -> ResponseCache:
    cache_config = dict(config.get("cache", {}))
    policy = str(cache_config.get("policy", "disabled"))
    enabled = bool(cache_config.get("enabled", False)) and policy != "disabled"
    return ResponseCache(
        cache_config.get("cache_dir", "outputs/cache/llm"),
        enabled=enabled,
        write_enabled=policy == "read_write",
    )


def _make_provider(
    llm_config: dict[str, Any],
    *,
    config: dict[str, Any],
    cache: ResponseCache,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url=str(llm_config["base_url"]),
        model=str(llm_config["model"]),
        api_key_env=str(llm_config.get("api_key_env", "OPENAI_API_KEY")),
        run_mode=str(config.get("experiment", {}).get("run_mode", "diagnostic_api")),
        allow_api_calls=bool(llm_config.get("allow_api_calls", False)),
        timeout_seconds=float(llm_config.get("timeout_seconds", 60.0)),
        max_retries=int(llm_config.get("max_retries", 2)),
        cache=cache,
    )


def _make_request(
    *,
    prompt: Any,
    config: dict[str, Any],
    llm_config: dict[str, Any],
    sample: dict[str, Any],
    prompt_variant: str,
    run_dir: Path,
) -> LLMRequest:
    spec = get_prompt_variant(prompt_variant)
    structured_config = dict(llm_config.get("structured_output", {}))
    metadata: dict[str, Any] = {
        "base_url_hash": _hash_base_url(str(llm_config.get("base_url", ""))),
        "dataset_run_id": config.get("source", {}).get("phase2b_run_dir"),
        "prompt_variant": prompt_variant,
        "run_dir": str(run_dir),
        "run_mode": config.get("experiment", {}).get("run_mode", "diagnostic_api"),
        "sample_group": sample.get("case_group", sample.get("group")),
        "sample_id": sample.get("sample_id"),
    }
    metadata.update(
        structured_output_metadata(
            enabled=bool(structured_config.get("enabled", True)),
            strict=bool(structured_config.get("strict", True)),
        )
    )
    if spec.uses_transition_evidence:
        metadata["transition_evidence"] = sample.get("transition_evidence", [])
        metadata["transition_scores"] = sample.get("transition_scores", {})
    if spec.uses_time_window_evidence:
        metadata["time_window_evidence"] = sample.get("time_window_evidence", [])
        metadata["time_window_scores"] = sample.get("time_window_scores", {})
    if spec.uses_time_buckets:
        metadata["time_bucket_scores"] = sample.get("transition_scores", {})
    return LLMRequest(
        prompt=prompt.prompt,
        prompt_version=prompt.prompt_version,
        candidate_item_ids=prompt.candidate_item_ids,
        provider=str(llm_config.get("provider", "openai_compatible")),
        model=str(llm_config.get("model")),
        decoding_params=dict(llm_config.get("decoding", {})),
        metadata=metadata,
    )


def _request_audit_row(
    request: LLMRequest,
    *,
    sample: dict[str, Any],
    cache: ResponseCache,
) -> dict[str, Any]:
    return {
        "cache_key": cache.key_for(request),
        "candidate_item_ids": request.candidate_item_ids,
        "decoding_params": request.decoding_params,
        "metadata": _safe_request_metadata(request.metadata),
        "model": request.model,
        "prompt": request.prompt,
        "prompt_variant": request.metadata.get("prompt_variant"),
        "prompt_version": request.prompt_version,
        "provider": request.provider,
        "sample_id": sample["sample_id"],
    }


def _safe_request_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    output = dict(metadata)
    output.pop("structured_output_schema", None)
    return output


def _hash_base_url(base_url: str) -> str:
    return hashlib.sha256(str(base_url).rstrip("/").encode("utf-8")).hexdigest()


def _load_completed_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        return []
    return read_jsonl(path)


def _prediction_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata", {})
    return (str(metadata.get("sample_id")), str(metadata.get("prompt_variant")))


def _metric_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return EMPTY_METRIC_FIELDS
    keys: set[str] = set(EMPTY_METRIC_FIELDS)
    for row in rows:
        keys.update(row)
    return sorted(keys)


def _grounding_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        rows.append(
            {
                "case_group": row.get("case_group"),
                "contrastive_evidence_usage_rate": row.get("contrastive_evidence_usage_rate", 0.0),
                "evidence_grounding_rate": row.get("evidence_grounding_rate", 0.0),
                "prompt_variant": row.get("prompt_variant"),
                "semantic_evidence_usage_rate": row.get("semantic_evidence_usage_rate", 0.0),
                "time_evidence_usage_rate": row.get("time_evidence_usage_rate", 0.0),
                "transition_evidence_usage_rate": row.get("transition_evidence_usage_rate", 0.0),
            }
        )
    return rows
