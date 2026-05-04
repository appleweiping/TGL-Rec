"""Phase 9D DeepSeek API LLM reranking experiment runner."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.evaluation.candidate_resolver import CandidateResolver
from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.io.artifacts import ensure_dir, iter_jsonl, sha256_file, write_csv_rows, write_json, write_jsonl
from llm4rec.llm.api_cache import APICache
from llm4rec.llm.api_cost import DEEPSEEK_V4_FLASH_PRICING, summarize_cost_latency
from llm4rec.llm.async_batch import run_async_batch
from llm4rec.llm.base import LLMRequest
from llm4rec.llm.cost_estimator import estimate_token_count
from llm4rec.llm.deepseek_provider import DeepSeekProviderConfig, DeepSeekV4FlashProvider
from llm4rec.llm.json_parser import LLMJSONParseError, parse_rerank_json
from llm4rec.llm.rate_limit import RateLimitConfig
from llm4rec.metrics.ranking import aggregate_ranking_metrics
from llm4rec.utils.env import collect_environment


PROMPT_VARIANTS = {
    "history_only": "deepseek_history_only_rerank",
    "history_with_order": "deepseek_order_rerank",
    "history_with_time_buckets": "deepseek_time_bucket_rerank",
    "history_with_transition_evidence": "deepseek_transition_evidence_rerank",
    "history_with_contrastive_evidence": "deepseek_contrastive_evidence_rerank",
    "time_graph_evidence_prompt": "deepseek_tge_evidence_rerank",
}


@dataclass(frozen=True)
class LLMCase:
    """One user/candidate prompt case."""

    dataset: str
    user_id: str
    target_item: str
    domain: str
    history: list[str]
    top_m_candidates: list[str]
    upstream_ranked_items: list[str]
    candidate_size: int
    target_included_top_m: bool
    candidate_ref: dict[str, Any]


@dataclass(frozen=True)
class PlannedRequest:
    """One planned DeepSeek reranking request."""

    case: LLMCase
    prompt_variant: str
    method: str
    prompt: str


def estimate_deepseek_cost(config_path: str | Path) -> dict[str, Any]:
    """Write a dry-run cost estimate without making API calls."""

    config = _load_config(config_path)
    run_dir = ensure_dir(resolve_path(config["experiment"]["dry_run_output_dir"]))
    plan = _build_plan(config, stage="dry_run")
    estimate = _cost_estimate(config, plan)
    write_json(run_dir / "cost_estimate.json", estimate)
    return estimate


def run_deepseek_matrix(config_path: str | Path, *, dry_run: bool = False, stage: str | None = None) -> dict[str, Any]:
    """Run or dry-run the configured Phase 9D DeepSeek matrix."""

    config = _load_config(config_path)
    selected_stage = stage or ("dry_run" if dry_run else str(config["experiment"].get("stage", "pilot")))
    run_dir = ensure_dir(resolve_path(config["experiment"][f"{selected_stage}_output_dir"]))
    plan = _build_plan(config, stage=selected_stage)
    _write_preflight_artifacts(config, plan, run_dir)
    if dry_run or selected_stage == "dry_run":
        return {"mode": "dry_run", "planned_requests": len(plan), "run_dir": str(run_dir)}
    if not os.environ.get(str(config["llm"]["api_key_env"])):
        raise RuntimeError(f"Missing API key environment variable: {config['llm']['api_key_env']}")
    return _execute_api_plan(config, plan, run_dir, selected_stage)


def export_deepseek_tables(run_dir: str | Path) -> dict[str, str]:
    """Export a separate API LLM comparison table."""

    root = resolve_path(run_dir)
    metrics_path = root / "metrics_by_method.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics_by_method.csv: {metrics_path}")
    rows = _read_csv(metrics_path)
    table_rows = _comparison_rows(rows)
    csv_path = root / "table_deepseek_llm.csv"
    tex_path = root / "table_deepseek_llm.tex"
    columns = [
        "dataset",
        "method",
        "Recall@5",
        "NDCG@5",
        "MRR@10",
        "parse_success_rate",
        "candidate_adherence_rate",
        "hallucination_rate",
        "estimated_cost",
        "latency_p95",
    ]
    write_csv_rows(csv_path, table_rows, fieldnames=columns)
    tex_path.write_text(_latex_table(table_rows, columns), encoding="utf-8", newline="\n")
    return {"table_csv": str(csv_path), "table_tex": str(tex_path)}


def _load_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    if "llm_config_path" in config:
        llm_config = load_yaml_config(config["llm_config_path"]).get("llm", {})
        config["llm"] = {**llm_config, **dict(config.get("llm", {}))}
    config.setdefault("experiment", {})
    config.setdefault("concurrency", {})
    config.setdefault("selection", {})
    config.setdefault("llm", {})
    return config


def _build_plan(config: dict[str, Any], *, stage: str) -> list[PlannedRequest]:
    datasets = list(config["datasets"])
    variants = [str(value) for value in config.get("prompt_variants", PROMPT_VARIANTS.keys())]
    max_users = _stage_limit(config, stage, "max_users_per_dataset")
    max_requests = _stage_limit(config, stage, "max_requests")
    cases_by_dataset: list[list[LLMCase]] = []
    for dataset in datasets:
        cases_by_dataset.append(_dataset_cases(config, str(dataset), max_users_per_dataset=max_users))
    plan: list[PlannedRequest] = []
    max_cases = max((len(cases) for cases in cases_by_dataset), default=0)
    for index in range(max_cases):
        for cases in cases_by_dataset:
            if index >= len(cases):
                continue
            case = cases[index]
            for variant in variants:
                if max_requests is not None and len(plan) >= int(max_requests):
                    return plan
                method = PROMPT_VARIANTS[str(variant)]
                plan.append(
                    PlannedRequest(
                        case=case,
                        prompt_variant=str(variant),
                        method=method,
                        prompt=_build_prompt(case, str(variant)),
                    )
                )
    return plan


def _dataset_cases(
    config: dict[str, Any],
    dataset: str,
    *,
    max_users_per_dataset: int | None,
) -> list[LLMCase]:
    dataset_cfg = dict(config["dataset_artifacts"][dataset])
    split_artifact = resolve_path(dataset_cfg["split_artifact"])
    candidate_artifact = resolve_path(dataset_cfg["candidate_artifact"])
    candidate_pool = dataset_cfg.get("candidate_pool_artifact")
    candidate_pool_path = resolve_path(candidate_pool) if candidate_pool else None
    candidate_ref_base = {
        "artifact_path": str(candidate_artifact),
        "artifact_sha256": sha256_file(candidate_artifact),
    }
    if candidate_pool_path and candidate_pool_path.is_file():
        candidate_ref_base["candidate_pool_artifact"] = str(candidate_pool_path)
        candidate_ref_base["candidate_pool_sha256"] = sha256_file(candidate_pool_path)
        candidate_ref_base["candidate_storage"] = "shared_pool"
    resolver = CandidateResolver(
        candidate_artifact_path=candidate_artifact,
        candidate_artifact_sha256=candidate_ref_base["artifact_sha256"],
        candidate_pool_path=candidate_pool_path,
        candidate_pool_sha256=candidate_ref_base.get("candidate_pool_sha256"),
    )
    histories, tests = _load_histories_and_tests(split_artifact)
    upstream = _load_upstream_predictions(config, dataset)
    top_m = int(config["selection"].get("top_m_candidates_for_llm", 50))
    cases: list[LLMCase] = []
    for row in tests:
        if max_users_per_dataset is not None and len(cases) >= int(max_users_per_dataset):
            break
        user_id = str(row["user_id"])
        target = str(row["item_id"])
        candidate_ref = dict(candidate_ref_base)
        candidate_ref["candidate_row_id"] = str(row.get("candidate_row_id", ""))
        candidates = resolver.get_candidates(
            user_id=user_id,
            target_item=target,
            candidate_ref=candidate_ref,
        )
        candidate_ref["candidate_size"] = len(candidates)
        selected = _select_top_m_candidates(candidates, upstream.get(user_id, []), top_m=top_m)
        cases.append(
            LLMCase(
                dataset=dataset,
                user_id=user_id,
                target_item=target,
                domain=str(row.get("domain") or dataset),
                history=histories.get(user_id, [])[-20:],
                top_m_candidates=selected,
                upstream_ranked_items=upstream.get(user_id, []),
                candidate_size=len(candidates),
                target_included_top_m=target in selected,
                candidate_ref=candidate_ref,
            )
        )
    return cases


def _load_histories_and_tests(split_artifact: Path) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    train_by_user: dict[str, list[tuple[float, str]]] = defaultdict(list)
    tests: list[dict[str, Any]] = []
    for row in iter_jsonl(split_artifact):
        split = str(row.get("split", ""))
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        if split == "train":
            timestamp = float(row.get("timestamp") or 0.0)
            train_by_user[user_id].append((timestamp, item_id))
        elif split == "test":
            tests.append(row)
    histories = {
        user: [item for _timestamp, item in sorted(items)]
        for user, items in train_by_user.items()
    }
    tests.sort(key=lambda row: str(row["user_id"]))
    return histories, tests


def _load_upstream_predictions(config: dict[str, Any], dataset: str) -> dict[str, list[str]]:
    root = resolve_path(config["selection"].get("upstream_run_dir", "outputs/paper_runs/protocol_v1/main_accuracy_multiseed/seed_0"))
    method = str(config["selection"].get("upstream_method", "time_graph_evidence"))
    path = root / dataset / method / "predictions.jsonl"
    if not path.is_file():
        fallback = resolve_path("outputs/paper_runs/protocol_v1/main_accuracy_seed0") / dataset / method / "predictions.jsonl"
        path = fallback if fallback.is_file() else path
    if not path.is_file():
        return {}
    output: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        output[str(row.get("user_id", ""))] = [str(item) for item in row.get("predicted_items", [])]
    return output


def _select_top_m_candidates(candidates: list[str], upstream_ranked: list[str], *, top_m: int) -> list[str]:
    candidate_set = {str(item) for item in candidates}
    selected: list[str] = []
    seen: set[str] = set()
    for item in upstream_ranked:
        item_id = str(item)
        if item_id in candidate_set and item_id not in seen:
            selected.append(item_id)
            seen.add(item_id)
        if len(selected) >= top_m:
            return selected
    for item in candidates:
        item_id = str(item)
        if item_id not in seen:
            selected.append(item_id)
            seen.add(item_id)
        if len(selected) >= top_m:
            break
    return selected


def _build_prompt(case: LLMCase, variant: str) -> str:
    evidence_lines: list[str] = []
    if variant in {"history_with_transition_evidence", "time_graph_evidence_prompt"}:
        evidence_lines.append("Transition evidence: prefer candidates that plausibly follow recent history items.")
    if variant in {"history_with_time_buckets", "time_graph_evidence_prompt"}:
        evidence_lines.append("Time evidence: recent interactions should usually matter more than old interactions.")
    if variant == "history_with_contrastive_evidence":
        evidence_lines.append("Contrastive evidence: separate semantic similarity from likely next-need transitions.")
    if variant == "time_graph_evidence_prompt":
        evidence_lines.append("Use temporal graph-to-language evidence when it conflicts with popularity.")
    history = case.history if variant != "history_only" else sorted(case.history)
    return (
        "Return json only. No markdown. No explanation. Rank at most 10 recommendation candidates. "
        "Use only IDs copied exactly from candidates.\n"
        f"Prompt variant: {variant}\n"
        f"User history: {history}\n"
        f"Candidates: {case.top_m_candidates}\n"
        + ("\n".join(evidence_lines) + "\n" if evidence_lines else "")
        + "Output this compact json shape with no extra keys: {\"ranked_item_ids\":[\"item_id\"], "
        "\"evidence_usage\": {\"transition\": true/false, \"time\": true/false, "
        "\"semantic\": true/false, \"contrastive\": true/false}}"
    )


def _write_preflight_artifacts(config: dict[str, Any], plan: list[PlannedRequest], run_dir: Path) -> None:
    estimate = _cost_estimate(config, plan)
    write_json(run_dir / "cost_estimate.json", estimate)
    write_jsonl(run_dir / "planned_requests.jsonl", (_planned_request_row(item) for item in plan))
    write_csv_rows(run_dir / "prompt_length_report.csv", _prompt_length_rows(plan))
    write_csv_rows(run_dir / "target_inclusion_audit.csv", _target_inclusion_rows(plan))
    write_json(run_dir / "api_safety_report.json", _api_safety_report(config, plan))


def _execute_api_plan(
    config: dict[str, Any],
    plan: list[PlannedRequest],
    run_dir: Path,
    stage: str,
) -> dict[str, Any]:
    save_resolved_config(_sanitized_config(config), run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment(resolve_path(".")))
    cache = APICache(
        run_dir / "cache",
        enabled=bool(config["llm"].get("cache_enabled", True)),
        resume=bool(config["llm"].get("resume", True)),
    )
    provider = DeepSeekV4FlashProvider(
        _provider_config(config),
        allow_api_calls=True,
        run_mode="diagnostic_api",
        cache=cache,
    )
    requests = [_to_llm_request(config, item) for item in plan]
    rate_config = _rate_limit_config(config)
    results, batch_report = asyncio.run(
        run_async_batch(
            requests,
            generate=provider.generate,
            rate_limit=rate_config,
            error_budget=int(config["experiment"].get("error_budget", 50)),
        )
    )
    request_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    hallucination_cases: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for result in results:
        planned = plan[result.index]
        request_rows.append(_api_request_row(planned, result.status, result.attempts))
        if result.response is None:
            parse_failures.append({**_request_identity(planned), "error": result.error, "stage": "api"})
            continue
        raw_rows.append({**_request_identity(planned), "raw_output": result.response.raw_output})
        usage_rows.append(
            {
                **_request_identity(planned),
                "cache_hit": result.response.cache_hit,
                "completion_tokens": result.response.completion_tokens,
                "latency_ms": result.response.latency_ms,
                "prompt_tokens": result.response.prompt_tokens,
                "total_tokens": result.response.total_tokens,
            }
        )
        try:
            parsed = parse_rerank_json(
                result.response.raw_output,
                candidate_item_ids=planned.case.top_m_candidates,
            )
        except LLMJSONParseError as exc:
            parse_failures.append({**_request_identity(planned), "error": str(exc), "stage": "parse"})
            continue
        predicted = _complete_prediction(parsed.ranked_item_ids, planned.case.top_m_candidates)
        row = {
            "candidate_items": planned.case.top_m_candidates,
            "domain": planned.case.domain,
            "method": planned.method,
            "metadata": {
                "dataset": planned.case.dataset,
                "evidence_usage": parsed.evidence_usage,
                "prompt_variant": planned.prompt_variant,
                "target_included_top_m": planned.case.target_included_top_m,
                "top_m_selection_policy": "upstream_predictions_then_frozen_candidates",
                "upstream_method": config["selection"].get("upstream_method", "time_graph_evidence"),
            },
            "predicted_items": predicted[:10],
            "raw_output": result.response.raw_output,
            "scores": [1.0 / float(index + 1) for index, _item in enumerate(predicted[:10])],
            "target_item": planned.case.target_item,
            "user_id": planned.case.user_id,
        }
        if parsed.invalid_item_ids:
            hallucination_cases.append({**_request_identity(planned), "invalid_item_ids": parsed.invalid_item_ids})
        prediction_rows.append(row)
    write_jsonl(run_dir / "api_requests.jsonl", request_rows)
    write_jsonl(run_dir / "api_raw_outputs.jsonl", raw_rows)
    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(run_dir / "parse_failures.jsonl", parse_failures)
    write_jsonl(run_dir / "hallucination_cases.jsonl", hallucination_cases)
    metrics = _metrics(
        prediction_rows,
        parse_failures=parse_failures,
        planned_count=len(plan),
        usage_rows=usage_rows,
        hallucination_cases=hallucination_cases,
    )
    write_json(run_dir / "metrics.json", metrics)
    write_csv_rows(run_dir / "metrics.csv", _metrics_long_rows(metrics))
    write_csv_rows(run_dir / "metrics_by_method.csv", _metrics_by_method_rows(metrics))
    cost_latency = summarize_cost_latency(usage_rows, pricing=DEEPSEEK_V4_FLASH_PRICING)
    write_json(run_dir / "cost_latency.json", cost_latency)
    write_json(run_dir / "cache_report.json", cache.report())
    write_json(run_dir / "failure_report.json", _failure_report(parse_failures, batch_report))
    write_csv_rows(run_dir / "grounding_report.csv", _grounding_rows(prediction_rows))
    export_deepseek_tables(run_dir)
    if stage == "pilot":
        _assert_pilot_acceptance(metrics)
    return {
        "batch_report": batch_report,
        "cost_latency": cost_latency,
        "metrics": metrics,
        "run_dir": str(run_dir),
    }


def _to_llm_request(config: dict[str, Any], planned: PlannedRequest) -> LLMRequest:
    llm = config["llm"]
    decoding = {
        "max_tokens": int(llm.get("max_tokens", 512)),
        "stream": bool(llm.get("stream", False)),
        "temperature": float(llm.get("temperature", 0.0)),
        "thinking": str(llm.get("thinking", "disabled")),
        "top_p": float(llm.get("top_p", 1.0)),
    }
    return LLMRequest(
        prompt=planned.prompt,
        prompt_version=planned.prompt_variant,
        candidate_item_ids=planned.case.top_m_candidates,
        provider="deepseek",
        model=str(llm.get("model", "deepseek-v4-flash")),
        decoding_params=decoding,
        metadata={"run_mode": "diagnostic_api"},
    )


def _provider_config(config: dict[str, Any]) -> DeepSeekProviderConfig:
    llm = config["llm"]
    return DeepSeekProviderConfig(
        base_url=str(llm.get("base_url", "https://api.deepseek.com")),
        model=str(llm.get("model", "deepseek-v4-flash")),
        api_key_env=str(llm.get("api_key_env", "DEEPSEEK_API_KEY")),
        temperature=float(llm.get("temperature", 0.0)),
        top_p=float(llm.get("top_p", 1.0)),
        max_tokens=int(llm.get("max_tokens", 512)),
        stream=bool(llm.get("stream", False)),
        thinking=str(llm.get("thinking", "disabled")),
        timeout=float(llm.get("timeout", 120)),
        max_retries=int(llm.get("max_retries", 8)),
        retry_on_status=tuple(int(value) for value in llm.get("retry_on_status", [429, 500, 502, 503, 504])),
    )


def _rate_limit_config(config: dict[str, Any]) -> RateLimitConfig:
    values = config.get("concurrency", {})
    llm = config.get("llm", {})
    return RateLimitConfig(
        max_concurrency=int(values.get("max_concurrency", 32)),
        adaptive_concurrency=bool(values.get("adaptive_concurrency", True)),
        min_concurrency=int(values.get("min_concurrency", 4)),
        max_concurrency_hard_cap=int(values.get("max_concurrency_hard_cap", 128)),
        backoff_initial_seconds=float(values.get("backoff_initial_seconds", 2)),
        backoff_max_seconds=float(values.get("backoff_max_seconds", 60)),
        jitter=bool(values.get("jitter", True)),
        max_retries=int(llm.get("max_retries", 8)),
        retry_on_status=tuple(int(value) for value in llm.get("retry_on_status", [429, 500, 502, 503, 504])),
    )


def _metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    parse_failures: list[dict[str, Any]],
    planned_count: int,
    usage_rows: list[dict[str, Any]],
    hallucination_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    planned_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in parse_failures:
        planned_counts[(str(row.get("dataset", "")), str(row.get("method", "")))] += 1
    for row in prediction_rows:
        planned_counts[(str(row.get("metadata", {}).get("dataset", "")), str(row.get("method", "")))] += 1
    invalid_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in hallucination_cases:
        invalid_counts[(str(row.get("dataset", "")), str(row.get("method", "")))] += len(
            row.get("invalid_item_ids", []) or []
        )
    by_dataset_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        dataset = str(row.get("metadata", {}).get("dataset", _dataset_from_domain(str(row.get("domain", "")))))
        by_dataset_method[(dataset, str(row["method"]))].append(row)
    method_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (dataset, method), rows in sorted(by_dataset_method.items()):
        ranking = aggregate_ranking_metrics(rows, ks=(1, 5, 10))
        usage = _evidence_usage_metrics(rows)
        method_metrics[dataset][method] = {
            **ranking,
            **usage,
            "candidate_adherence_rate": _candidate_adherence_rate(
                rows,
                invalid_count=invalid_counts.get((dataset, method), 0),
            ),
            "evidence_grounding_rate": usage["transition_evidence_usage_rate"],
            "hallucination_rate": _hallucination_rate(
                rows,
                invalid_count=invalid_counts.get((dataset, method), 0),
            ),
            "num_predictions": float(len(rows)),
            "parse_success_rate": (len(rows) / float(planned_counts.get((dataset, method), planned_count) or 1)),
            "validity_rate": 1.0
            - _hallucination_rate(rows, invalid_count=invalid_counts.get((dataset, method), 0)),
        }
    return {
        "by_dataset_method": method_metrics,
        "cost_latency": summarize_cost_latency(usage_rows, pricing=DEEPSEEK_V4_FLASH_PRICING),
        "num_parse_failures": len(parse_failures),
        "num_predictions": len(prediction_rows),
        "planned_requests": planned_count,
    }


def _dataset_from_domain(domain: str) -> str:
    return "movielens_full" if domain == "movielens" else "amazon_multidomain_filtered_iterative_k3"


def _candidate_adherence_rate(rows: list[dict[str, Any]], *, invalid_count: int = 0) -> float:
    total = 0
    adherent = 0
    for row in rows:
        candidates = {str(item) for item in row.get("candidate_items", [])}
        for item in row.get("predicted_items", []):
            total += 1
            if str(item) in candidates:
                adherent += 1
    return adherent / float(total + int(invalid_count) or 1)


def _hallucination_rate(rows: list[dict[str, Any]], *, invalid_count: int = 0) -> float:
    total = sum(len(row.get("predicted_items", []) or []) for row in rows) + int(invalid_count)
    return int(invalid_count) / float(total or 1)


def _evidence_usage_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = defaultdict(float)
    for row in rows:
        usage = row.get("metadata", {}).get("evidence_usage", {})
        for key in ("transition", "time", "semantic", "contrastive"):
            totals[key] += 1.0 if usage.get(key) else 0.0
    denom = float(len(rows) or 1)
    return {
        "contrastive_evidence_usage_rate": totals["contrastive"] / denom,
        "semantic_evidence_usage_rate": totals["semantic"] / denom,
        "time_evidence_usage_rate": totals["time"] / denom,
        "transition_evidence_usage_rate": totals["transition"] / denom,
    }


def _complete_prediction(parsed_items: list[str], candidates: list[str]) -> list[str]:
    output = list(parsed_items)
    seen = set(output)
    for item in candidates:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _cost_estimate(config: dict[str, Any], plan: list[PlannedRequest]) -> dict[str, Any]:
    max_tokens = int(config["llm"].get("max_tokens", 512))
    prompt_tokens = sum(estimate_token_count(item.prompt) for item in plan)
    completion_tokens = max_tokens * len(plan)
    estimated_cost = (
        prompt_tokens / 1_000_000.0 * DEEPSEEK_V4_FLASH_PRICING.input_cache_miss_per_1m
        + completion_tokens / 1_000_000.0 * DEEPSEEK_V4_FLASH_PRICING.output_per_1m
    )
    return {
        "estimated_completion_tokens": completion_tokens,
        "estimated_cost_usd_cache_miss": estimated_cost,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_requests": len(plan),
        "max_tokens": max_tokens,
        "model": config["llm"].get("model", "deepseek-v4-flash"),
        "pricing_source": DEEPSEEK_V4_FLASH_PRICING.source,
        "shared_pool_scoring_note": "shared-pool scoring is an efficiency fix, not a protocol change.",
    }


def _stage_limit(config: dict[str, Any], stage: str, key: str) -> int | None:
    section = config["experiment"].get(stage, {})
    value = section.get(key, config["experiment"].get(key))
    return None if value in (None, "null") else int(value)


def _planned_request_row(item: PlannedRequest) -> dict[str, Any]:
    return {
        **_request_identity(item),
        "candidate_size": item.case.candidate_size,
        "prompt_sha256": _sha256_text(item.prompt),
        "prompt_tokens_estimate": estimate_token_count(item.prompt),
        "target_included_top_m": item.case.target_included_top_m,
        "top_m_candidates": len(item.case.top_m_candidates),
    }


def _request_identity(item: PlannedRequest) -> dict[str, Any]:
    return {
        "dataset": item.case.dataset,
        "method": item.method,
        "prompt_variant": item.prompt_variant,
        "target_item": item.case.target_item,
        "user_id": item.case.user_id,
    }


def _api_request_row(item: PlannedRequest, status: str, attempts: int) -> dict[str, Any]:
    return {
        **_planned_request_row(item),
        "attempts": attempts,
        "status": status,
    }


def _prompt_length_rows(plan: list[PlannedRequest]) -> list[dict[str, Any]]:
    return [
        {
            **_request_identity(item),
            "characters": len(item.prompt),
            "prompt_tokens_estimate": estimate_token_count(item.prompt),
        }
        for item in plan
    ]


def _target_inclusion_rows(plan: list[PlannedRequest]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for item in plan:
        key = (item.case.dataset, item.case.user_id, item.case.target_item)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "candidate_size": item.case.candidate_size,
                "dataset": item.case.dataset,
                "target_included_top_m": item.case.target_included_top_m,
                "target_item": item.case.target_item,
                "top_m_candidates": len(item.case.top_m_candidates),
                "user_id": item.case.user_id,
            }
        )
    return rows


def _api_safety_report(config: dict[str, Any], plan: list[PlannedRequest]) -> dict[str, Any]:
    return {
        "api_key_env": config["llm"].get("api_key_env", "DEEPSEEK_API_KEY"),
        "api_key_value_saved": False,
        "authorization_header_saved": False,
        "llm_provider": "deepseek",
        "planned_requests": len(plan),
        "raw_secret_env_saved": False,
        "stream": bool(config["llm"].get("stream", False)),
    }


def _sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config))


def _failure_report(parse_failures: list[dict[str, Any]], batch_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "blocks_paper_scale_readiness": bool(parse_failures),
        "failure_count": len(parse_failures),
        "failures_sample": parse_failures[:20],
        "batch_report": batch_report,
    }


def _grounding_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        usage = row.get("metadata", {}).get("evidence_usage", {})
        rows.append(
            {
                "candidate_adherent": all(item in set(row.get("candidate_items", [])) for item in row.get("predicted_items", [])),
                "contrastive": usage.get("contrastive", False),
                "method": row.get("method", ""),
                "semantic": usage.get("semantic", False),
                "target_included_top_m": row.get("metadata", {}).get("target_included_top_m", False),
                "time": usage.get("time", False),
                "transition": usage.get("transition", False),
                "user_id": row.get("user_id", ""),
            }
        )
    return rows


def _metrics_long_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, method_payload in metrics.get("by_dataset_method", {}).items():
        for method, values in method_payload.items():
            for metric, value in values.items():
                rows.append({"dataset": dataset, "method": method, "metric": metric, "value": value})
    return rows


def _metrics_by_method_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cost_latency = metrics.get("cost_latency", {})
    for dataset, method_payload in metrics.get("by_dataset_method", {}).items():
        for method, values in method_payload.items():
            row = {"dataset": dataset, "method": method, **values}
            row.update({key: cost_latency.get(key, 0.0) for key in ("estimated_cost", "latency_p95")})
            rows.append(row)
    return rows


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "deepseek_history_only_rerank",
        "deepseek_transition_evidence_rerank",
        "deepseek_contrastive_evidence_rerank",
        "deepseek_tge_evidence_rerank",
    }
    output = [row for row in rows if str(row.get("method", "")) in wanted]
    baseline_path = resolve_path("outputs/paper_runs/protocol_v1/main_accuracy_multiseed/aggregate_metrics.csv")
    if baseline_path.is_file():
        output.extend(_baseline_rows(_read_csv(baseline_path)))
    return sorted(output, key=lambda row: (str(row.get("dataset", "")), str(row.get("method", ""))))


def _baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({str(row.get("dataset", "")) for row in rows})
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        for method in ("time_graph_evidence", _best_non_llm_method(dataset_rows)):
            if not method:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "Recall@5": _metric_value(dataset_rows, method, "Recall@5"),
                    "NDCG@5": _metric_value(dataset_rows, method, "NDCG@5"),
                    "MRR@10": _metric_value(dataset_rows, method, "MRR@10"),
                    "parse_success_rate": "",
                    "candidate_adherence_rate": "",
                    "hallucination_rate": "",
                    "estimated_cost": "",
                    "latency_p95": "",
                }
            )
    return output


def _best_non_llm_method(rows: list[dict[str, Any]]) -> str:
    excluded = {"time_graph_evidence", "time_graph_evidence_dynamic"}
    recall_rows = [row for row in rows if row.get("metric") == "Recall@5" and row.get("method") not in excluded]
    if not recall_rows:
        return ""
    return str(max(recall_rows, key=lambda row: float(row.get("mean", 0.0) or 0.0)).get("method", ""))


def _metric_value(rows: list[dict[str, Any]], method: str, metric: str) -> str:
    for row in rows:
        if row.get("method") == method and row.get("metric") == metric:
            return str(row.get("mean", ""))
    return ""


def _latex_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(_latex(column) for column in columns) + " \\\\"]
    for row in rows:
        lines.append(" & ".join(_latex(row.get(column, "")) for column in columns) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _latex(value: Any) -> str:
    text = str(value)
    return text.replace("_", "\\_")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _assert_pilot_acceptance(metrics: dict[str, Any]) -> None:
    values = [
        float(method_values.get("parse_success_rate", 0.0))
        for dataset_values in metrics.get("by_dataset_method", {}).values()
        for method_values in dataset_values.values()
    ]
    if values and min(values) < 0.95:
        raise RuntimeError(f"Pilot parse_success_rate below acceptance threshold: {min(values):.4f}")


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
