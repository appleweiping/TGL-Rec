"""Phase 3A LLM sequence/time diagnostic runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm4rec.data.time_features import consecutive_time_gaps
from llm4rec.diagnostics.diagnostic_export import run_phase2b_from_config
from llm4rec.diagnostics.llm_grounding import build_edge_index, evaluate_evidence_grounding
from llm4rec.evaluation.llm_diagnostic_evaluator import evaluate_llm_diagnostic_predictions
from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.cost_tracker import CostLatencyTracker
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.llm.response_cache import ResponseCache
from llm4rec.prompts.base import PromptExample
from llm4rec.prompts.builder import build_prompt
from llm4rec.prompts.parsers import try_parse_llm_response
from llm4rec.prompts.variants import get_prompt_variant
from llm4rec.utils.env import collect_environment


DEFAULT_PROMPT_VARIANTS = [
    "history_only",
    "history_with_order",
    "history_with_time_gaps",
    "history_with_time_buckets",
    "history_with_transition_evidence",
    "history_with_time_window_evidence",
    "history_with_contrastive_evidence",
]


def run_llm_sequence_time_diagnostics(config_path: str | Path) -> Path:
    """Run fixed-sample LLM prompt diagnostics with mock or explicit API provider."""

    config = load_yaml_config(config_path)
    experiment = dict(config.get("experiment", {}))
    seed = int(experiment.get("seed", 2026))
    set_global_seed(seed)
    run_id = str(experiment.get("run_id", "phase3a_llm_diagnostics"))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = ensure_dir(output_root / run_id)
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")
    ensure_dir(run_dir / "artifacts")
    logger = RunLogger(run_dir / "logs.txt")
    logger.info(f"starting Phase 3A LLM diagnostics run_id={run_id}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())

    source_run_dir = _prepare_source_run(config, logger)
    processed_dir = source_run_dir / "artifacts" / "processed_dataset"
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    item_records = {str(row["item_id"]): row for row in item_rows}
    item_catalog = set(item_records)
    interactions = read_jsonl(processed_dir / "interactions.jsonl")
    candidate_rows = [row for row in read_jsonl(processed_dir / "candidates.jsonl") if row.get("split") == "test"]
    grouped_cases = _load_grouped_cases(source_run_dir)
    transition_edges = read_jsonl(source_run_dir / "diagnostics" / "transition_edges.jsonl")
    time_window_edges = _load_time_window_edges(source_run_dir)
    transition_index = build_edge_index(transition_edges)
    time_window_index = build_edge_index(time_window_edges)

    diagnostic = dict(config.get("diagnostic", {}))
    prompt_variants = [str(name) for name in diagnostic.get("prompt_variants", DEFAULT_PROMPT_VARIANTS)]
    max_cases_per_group = int(diagnostic.get("max_cases_per_group", 2))
    candidate_size = int(diagnostic.get("candidate_size", 8))
    samples = _build_diagnostic_samples(
        grouped_cases=grouped_cases,
        candidate_rows=candidate_rows,
        interactions=interactions,
        item_records=item_records,
        max_cases_per_group=max_cases_per_group,
        candidate_size=candidate_size,
    )
    if not samples:
        raise ValueError("No Phase 3A diagnostic samples could be built from Phase 2B artifacts.")
    write_json(run_dir / "artifacts" / "diagnostic_samples.json", {"samples": samples})

    llm_config = _load_llm_config(config)
    provider_type = str(llm_config.get("provider", "mock"))
    run_mode = str(experiment.get("run_mode", llm_config.get("run_mode", "diagnostic_mock")))
    cache_config = dict(config.get("cache", {}))
    cache = ResponseCache(
        cache_config.get("cache_dir", "outputs/cache/llm"),
        enabled=bool(cache_config.get("enabled", True)),
    )
    tracker = CostLatencyTracker(pricing=dict(llm_config.get("pricing", {})))
    predictions: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    raw_output_rows: list[dict[str, Any]] = []
    hallucination_rows: list[dict[str, Any]] = []
    parse_failure_rows: list[dict[str, Any]] = []

    for sample in samples:
        example = _prompt_example_from_sample(sample, item_records)
        for prompt_variant in prompt_variants:
            prompt = build_prompt(
                example,
                prompt_variant=prompt_variant,
                max_history_items=int(diagnostic.get("max_history_items", 20)),
                max_evidence_items=int(diagnostic.get("max_evidence_items", 8)),
            )
            provider = _make_provider(
                llm_config,
                prompt_variant=prompt_variant,
                run_mode=run_mode,
                cache=None,
            )
            request = _make_request(
                prompt=prompt,
                config=config,
                llm_config=llm_config,
                run_mode=run_mode,
                sample=sample,
                provider_type=provider_type,
                prompt_variant=prompt_variant,
            )
            request_rows.append(_request_audit_row(request, sample_id=str(sample["sample_id"])))
            response = _cached_generate(cache, provider, request)
            tracker.record(response)
            raw_output_rows.append(
                {
                    "cache_hit": response.cache_hit,
                    "model": response.model,
                    "prompt_variant": prompt_variant,
                    "provider": response.provider,
                    "raw_output": response.raw_output,
                    "sample_id": sample["sample_id"],
                    "usage": _response_usage(response),
                }
            )
            parsed = try_parse_llm_response(response.raw_output, candidate_items=sample["candidate_items"])
            grounding = evaluate_evidence_grounding(
                parsed.evidence_used,
                history_items=sample["history"],
                candidate_items=sample["candidate_items"],
                transition_edges=transition_index,
                time_window_edges=time_window_index,
                time_bucket_by_pair=_parse_time_bucket_pairs(sample.get("time_bucket_by_pair", {})),
            )
            if not parsed.parse_success:
                parse_failure_rows.append(
                    {
                        "parse_error": parsed.metadata.get("parse_error"),
                        "prompt_variant": prompt_variant,
                        "raw_output": response.raw_output,
                        "sample_id": sample["sample_id"],
                    }
                )
            if parsed.invalid_item_ids:
                hallucination_rows.append(
                    {
                        "invalid_item_ids": parsed.invalid_item_ids,
                        "prompt_variant": prompt_variant,
                        "raw_output": response.raw_output,
                        "sample_id": sample["sample_id"],
                    }
                )
            predictions.append(
                {
                    "candidate_items": sample["candidate_items"],
                    "domain": sample.get("domain"),
                    "metadata": {
                        "duplicate_item_ids": parsed.duplicate_item_ids,
                        "grounding": grounding,
                        "invalid_item_ids": parsed.invalid_item_ids,
                        "llm_usage": _response_usage(response),
                        "parse_error": parsed.metadata.get("parse_error"),
                        "parse_success": parsed.parse_success,
                        "phase": "phase3a",
                        "prompt_variant": prompt_variant,
                        "prompt_version": prompt.prompt_version,
                        "reasoning_summary": parsed.reasoning_summary,
                        "run_mode": run_mode,
                        "sample_group": sample.get("group"),
                        "sample_id": sample["sample_id"],
                    },
                    "method": "llm_rerank_diagnostic",
                    "predicted_items": parsed.ranked_item_ids + parsed.invalid_item_ids,
                    "raw_output": response.raw_output,
                    "scores": [float(len(parsed.ranked_item_ids) - index) for index, _ in enumerate(parsed.ranked_item_ids)]
                    + [0.0 for _ in parsed.invalid_item_ids],
                    "target_item": sample["target_item"],
                    "user_id": sample["user_id"],
                }
            )

    write_jsonl(run_dir / "llm_requests.jsonl", request_rows)
    write_jsonl(run_dir / "llm_raw_outputs.jsonl", raw_output_rows)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(diagnostics_dir / "hallucination_cases.jsonl", hallucination_rows)
    write_jsonl(diagnostics_dir / "parse_failures.jsonl", parse_failure_rows)

    evaluation = dict(config.get("evaluation", {}))
    ks = tuple(int(k) for k in evaluation.get("ks", [1, 3, 5]))
    candidate_protocol = str(evaluation.get("candidate_protocol", "full_catalog"))
    top_k = int(evaluation.get("top_k", 5))
    metric_rows, delta_rows, overlap_rows = evaluate_llm_diagnostic_predictions(
        prediction_rows=predictions,
        item_catalog=item_catalog,
        ks=ks,
        candidate_protocol=candidate_protocol,
        top_k=top_k,
    )
    grounding_rows = _grounding_rows(metric_rows)
    write_json(run_dir / "metrics.json", {"metrics": metric_rows})
    write_csv_rows(run_dir / "metrics.csv", metric_rows)
    write_csv_rows(diagnostics_dir / "prompt_variant_results.csv", metric_rows)
    write_csv_rows(diagnostics_dir / "prompt_variant_deltas.csv", delta_rows)
    write_csv_rows(diagnostics_dir / "prediction_overlap_by_prompt.csv", overlap_rows)
    write_csv_rows(diagnostics_dir / "grounding_summary.csv", grounding_rows)
    cost_latency = tracker.summary()
    if provider_type == "openai_compatible" and run_mode != "diagnostic_api":
        cost_latency["api_status"] = "disabled"
    write_json(run_dir / "cost_latency.json", cost_latency)
    summary = build_llm_diagnostic_summary(
        metric_rows=metric_rows,
        overlap_rows=overlap_rows,
        grounding_rows=grounding_rows,
        hallucination_rows=hallucination_rows,
        parse_failure_rows=parse_failure_rows,
        cost_latency=cost_latency,
    )
    write_json(run_dir / "llm_diagnostic_summary.json", summary)
    logger.info("Phase 3A LLM diagnostics completed")
    return run_dir


def export_llm_diagnostics(run_dir: str | Path) -> dict[str, Any]:
    """Rebuild the top-level LLM diagnostic summary from existing outputs."""

    root = Path(run_dir)
    metric_rows = _read_csv(root / "diagnostics" / "prompt_variant_results.csv")
    overlap_rows = _read_csv(root / "diagnostics" / "prediction_overlap_by_prompt.csv")
    grounding_rows = _read_csv(root / "diagnostics" / "grounding_summary.csv")
    hallucination_rows = read_jsonl(root / "diagnostics" / "hallucination_cases.jsonl")
    parse_failure_rows = read_jsonl(root / "diagnostics" / "parse_failures.jsonl")
    cost_latency = json.loads((root / "cost_latency.json").read_text(encoding="utf-8"))
    summary = build_llm_diagnostic_summary(
        metric_rows=metric_rows,
        overlap_rows=overlap_rows,
        grounding_rows=grounding_rows,
        hallucination_rows=hallucination_rows,
        parse_failure_rows=parse_failure_rows,
        cost_latency=cost_latency,
    )
    write_json(root / "llm_diagnostic_summary.json", summary)
    return summary


def build_llm_diagnostic_summary(
    *,
    metric_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    grounding_rows: list[dict[str, Any]],
    hallucination_rows: list[dict[str, Any]],
    parse_failure_rows: list[dict[str, Any]],
    cost_latency: dict[str, Any],
) -> dict[str, Any]:
    """Build numeric summary answers without paper-level claims."""

    overlaps = [
        float(row.get("prediction_overlap_vs_history_only@K", row.get("mean_prediction_overlap", 0.0)) or 0.0)
        for row in overlap_rows
    ]
    parse_by_prompt = {
        str(row["prompt_variant"]): float(row.get("parse_success_rate", 0.0) or 0.0)
        for row in metric_rows
    }
    validity_by_prompt = {
        str(row["prompt_variant"]): float(row.get("validity_rate", 0.0) or 0.0)
        for row in metric_rows
    }
    hallucination_by_prompt = {
        str(row["prompt_variant"]): float(row.get("hallucination_rate", 0.0) or 0.0)
        for row in metric_rows
    }
    return {
        "cost_latency": cost_latency,
        "grounding": {
            "evidence_grounding_rate": _mean(grounding_rows, "evidence_grounding_rate"),
            "semantic_evidence_usage_rate": _mean(grounding_rows, "semantic_evidence_usage_rate"),
            "time_evidence_usage_rate": _mean(grounding_rows, "time_evidence_usage_rate"),
            "transition_evidence_usage_rate": _mean(grounding_rows, "transition_evidence_usage_rate"),
        },
        "hallucination_case_count": len(hallucination_rows),
        "hallucination_rate_by_prompt": hallucination_by_prompt,
        "output_change_rate_vs_history_only": {
            str(row.get("prompt_variant")): float(row.get("output_change_rate", 0.0) or 0.0)
            for row in overlap_rows
        },
        "parse_failure_count": len(parse_failure_rows),
        "parse_success_rate_by_prompt": parse_by_prompt,
        "prediction_overlap_range": {
            "max": max(overlaps) if overlaps else 1.0,
            "min": min(overlaps) if overlaps else 1.0,
        },
        "prompt_variants_run": [str(row["prompt_variant"]) for row in metric_rows],
        "validity_rate_by_prompt": validity_by_prompt,
    }


def _prepare_source_run(config: dict[str, Any], logger: RunLogger) -> Path:
    source = dict(config.get("source", {}))
    run_dir = resolve_path(source.get("phase2b_run_dir", "outputs/runs/phase2b_movielens_diagnostics"))
    processed_dir = run_dir / "artifacts" / "processed_dataset"
    if processed_dir.is_dir() and (run_dir / "diagnostics" / "grouped_cases.json").is_file():
        return run_dir
    if bool(source.get("prepare_phase2b_if_missing", False)):
        phase2b_config = source.get("phase2b_config", "configs/experiments/phase2b_movielens_diagnostics.yaml")
        logger.info(f"Phase 2B source artifacts missing; preparing via {phase2b_config}")
        return run_phase2b_from_config(resolve_path(phase2b_config))
    raise FileNotFoundError(
        f"Missing Phase 2B artifacts at {run_dir}. Run Phase 2B first or set prepare_phase2b_if_missing=true."
    )


def _load_grouped_cases(source_run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = source_run_dir / "diagnostics" / "grouped_cases.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    similarity = json.loads((source_run_dir / "similarity_vs_transition.json").read_text(encoding="utf-8"))
    return dict(similarity.get("grouped_cases", {}))


def _load_time_window_edges(source_run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ("1h", "1d", "7d", "30d"):
        path = source_run_dir / "diagnostics" / f"time_window_edges_{label}.jsonl"
        if path.is_file():
            rows.extend(read_jsonl(path))
    return rows


def _build_diagnostic_samples(
    *,
    grouped_cases: dict[str, list[dict[str, Any]]],
    candidate_rows: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    item_records: dict[str, dict[str, Any]],
    max_cases_per_group: int,
    candidate_size: int,
) -> list[dict[str, Any]]:
    by_user = _interactions_by_user(interactions)
    examples = [_candidate_example(row, by_user) for row in candidate_rows]
    source_to_examples: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        for item_id in example["history"]:
            source_to_examples.setdefault(item_id, []).append(example)
    selected_cases = _select_group_cases(grouped_cases, max_cases_per_group=max_cases_per_group)
    samples: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()
    for group, case in selected_cases:
        source = str(case.get("source_item"))
        target = str(case.get("target_item"))
        candidate_example = _find_example(source_to_examples, examples, source=source, target=target)
        if candidate_example is None:
            continue
        key = (group, candidate_example["user_id"], target)
        if key in used:
            continue
        used.add(key)
        candidates = _candidate_subset(
            candidate_example["candidate_items"],
            required=[candidate_example["target_item"], target, source],
            item_records=item_records,
            candidate_size=candidate_size,
        )
        transition_evidence = [_normalize_case_evidence(case)]
        time_window_evidence = [_normalize_case_evidence(case)]
        time_bucket_by_pair = {
            f"{case.get('source_item')}->{case.get('target_item')}": str(case.get("dominant_gap_bucket", ""))
        }
        samples.append(
            {
                **candidate_example,
                "candidate_items": candidates,
                "contrastive_evidence": [_normalize_case_evidence(case)],
                "group": group,
                "sample_id": f"{group}__{candidate_example['user_id']}__{source}__{target}",
                "time_bucket_by_pair": time_bucket_by_pair,
                "time_window_evidence": time_window_evidence,
                "time_window_scores": _scores_from_case(case, prefix="time_window"),
                "transition_evidence": transition_evidence,
                "transition_scores": {target: float(case.get("transition_score", case.get("transition_count", 0.0)) or 0.0)},
            }
        )
    return samples


def _interactions_by_user(interactions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in interactions:
        by_user.setdefault(str(row["user_id"]), []).append(row)
    for rows in by_user.values():
        rows.sort(
            key=lambda row: (
                float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                str(row["item_id"]),
            )
        )
    return by_user


def _candidate_example(row: dict[str, Any], by_user: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    user_id = str(row["user_id"])
    target = str(row["target_item"])
    target_rows = [
        value
        for value in by_user.get(user_id, [])
        if str(value.get("item_id")) == target and str(value.get("split")) == str(row.get("split", "test"))
    ]
    target_ts = None
    if target_rows:
        target_ts = float(target_rows[-1]["timestamp"]) if target_rows[-1].get("timestamp") is not None else None
    history_rows = []
    for value in by_user.get(user_id, []):
        timestamp = None if value.get("timestamp") is None else float(value["timestamp"])
        if target_ts is not None and timestamp is not None and timestamp >= target_ts:
            continue
        if str(value.get("split")) == "test":
            continue
        history_rows.append(value)
    return {
        "candidate_items": [str(item) for item in row["candidate_items"]],
        "domain": row.get("domain"),
        "history": [str(value["item_id"]) for value in history_rows],
        "history_rows": history_rows,
        "target_item": target,
        "user_id": user_id,
    }


def _select_group_cases(
    grouped_cases: dict[str, list[dict[str, Any]]],
    *,
    max_cases_per_group: int,
) -> list[tuple[str, dict[str, Any]]]:
    groups = [
        "semantic_and_transition",
        "semantic_only",
        "transition_only",
        "cross_category_transition",
        "same_category_transition",
    ]
    output: list[tuple[str, dict[str, Any]]] = []
    for group in groups:
        rows = sorted(
            grouped_cases.get(group, []),
            key=lambda row: (
                -float(row.get("transition_score", row.get("transition_count", 0.0)) or 0.0),
                -float(row.get("time_window_score_1d", 0.0) or 0.0),
                str(row.get("source_item")),
                str(row.get("target_item")),
            ),
        )
        output.extend((group, row) for row in rows[:max_cases_per_group])
    return output


def _find_example(
    source_to_examples: dict[str, list[dict[str, Any]]],
    examples: list[dict[str, Any]],
    *,
    source: str,
    target: str,
) -> dict[str, Any] | None:
    for example in source_to_examples.get(source, []):
        if target in example["candidate_items"]:
            return example
    for example in examples:
        if target in example["candidate_items"]:
            return example
    return examples[0] if examples else None


def _candidate_subset(
    candidates: list[str],
    *,
    required: list[str],
    item_records: dict[str, dict[str, Any]],
    candidate_size: int,
) -> list[str]:
    output: list[str] = []
    for item_id in required:
        if item_id in item_records and item_id not in output:
            output.append(item_id)
    for item_id in candidates:
        if item_id in item_records and item_id not in output:
            output.append(item_id)
        if len(output) >= candidate_size:
            break
    return output[:candidate_size]


def _normalize_case_evidence(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "bucket_counts": {},
        "dominant_gap_bucket": case.get("dominant_gap_bucket"),
        "mean_time_gap": case.get("mean_time_gap"),
        "median_time_gap": case.get("median_time_gap"),
        "primary_group": case.get("primary_group"),
        "same_genre_or_category": case.get("same_genre_or_category"),
        "source_item": str(case.get("source_item")),
        "source_title": case.get("source_title"),
        "target_item": str(case.get("target_item")),
        "target_title": case.get("target_title"),
        "text_similarity": case.get("text_similarity"),
        "time_window_score_1d": case.get("time_window_score_1d", 0.0),
        "time_window_score_30d": case.get("time_window_score_30d", 0.0),
        "time_window_score_7d": case.get("time_window_score_7d", 0.0),
        "transition_count": case.get("transition_count", 0.0),
        "transition_score": case.get("transition_score", 0.0),
    }


def _scores_from_case(case: dict[str, Any], *, prefix: str) -> dict[str, float]:
    target = str(case.get("target_item"))
    if prefix == "time_window":
        return {
            target: max(
                float(case.get("time_window_score_1d", 0.0) or 0.0),
                float(case.get("time_window_score_7d", 0.0) or 0.0),
                float(case.get("time_window_score_30d", 0.0) or 0.0),
            )
        }
    return {target: float(case.get("transition_score", 0.0) or 0.0)}


def _parse_time_bucket_pairs(value: Any) -> dict[tuple[str, str], str]:
    if not isinstance(value, dict):
        return {}
    output: dict[tuple[str, str], str] = {}
    for key, bucket in value.items():
        if isinstance(key, tuple) and len(key) == 2:
            output[(str(key[0]), str(key[1]))] = str(bucket)
        elif isinstance(key, str) and "->" in key:
            source, target = key.split("->", 1)
            output[(source, target)] = str(bucket)
    return output


def _prompt_example_from_sample(
    sample: dict[str, Any],
    item_records: dict[str, dict[str, Any]],
) -> PromptExample:
    sample_items = set(sample["candidate_items"]) | set(sample["history"])
    records = {item_id: item_records[item_id] for item_id in sample_items if item_id in item_records}
    return PromptExample(
        user_id=str(sample["user_id"]),
        history=[str(item) for item in sample["history"]],
        target_item=str(sample["target_item"]),
        candidate_items=[str(item) for item in sample["candidate_items"]],
        domain=sample.get("domain"),
        item_records=records,
        history_rows=list(sample.get("history_rows", [])),
        transition_evidence=list(sample.get("transition_evidence", [])),
        time_window_evidence=list(sample.get("time_window_evidence", [])),
        contrastive_evidence=list(sample.get("contrastive_evidence", [])),
        metadata=sample,
    )


def _load_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    section = dict(config.get("llm", {}))
    if section.get("config_path"):
        loaded = load_yaml_config(section["config_path"])
        merged = {**dict(loaded.get("llm", loaded)), **{k: v for k, v in section.items() if k != "config_path"}}
        return merged
    return section


def _make_provider(
    llm_config: dict[str, Any],
    *,
    prompt_variant: str,
    run_mode: str,
    cache: ResponseCache | None,
) -> Any:
    provider = str(llm_config.get("provider", "mock"))
    if provider == "mock":
        modes = dict(llm_config.get("modes_by_prompt_variant", {}))
        mode = str(modes.get(prompt_variant, llm_config.get("mode", "identity")))
        return MockLLMProvider(
            mode=mode,
            model=str(llm_config.get("model", "mock-llm")),
            run_mode=run_mode,
        )
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            base_url=str(llm_config["base_url"]),
            model=str(llm_config["model"]),
            api_key_env=str(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            run_mode=run_mode,
            allow_api_calls=bool(llm_config.get("allow_api_calls", False)),
            timeout_seconds=float(llm_config.get("timeout_seconds", 60.0)),
            max_retries=int(llm_config.get("max_retries", 2)),
            cache=cache,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _make_request(
    *,
    prompt: Any,
    config: dict[str, Any],
    llm_config: dict[str, Any],
    run_mode: str,
    sample: dict[str, Any],
    provider_type: str,
    prompt_variant: str,
) -> LLMRequest:
    spec = get_prompt_variant(prompt_variant)
    metadata: dict[str, Any] = {
        "dataset_run_id": config.get("source", {}).get("phase2b_run_dir"),
        "run_mode": run_mode,
        "sample_group": sample.get("group"),
        "sample_id": sample.get("sample_id"),
    }
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
        provider=provider_type,
        model=str(llm_config.get("model", "mock-llm")),
        decoding_params=dict(llm_config.get("decoding", {})),
        metadata=metadata,
    )


def _cached_generate(cache: ResponseCache, provider: Any, request: LLMRequest) -> LLMResponse:
    cached = cache.get(request)
    if cached is not None:
        return cached
    response = provider.generate(request)
    cache.set(request, response)
    return response


def _request_audit_row(request: LLMRequest, *, sample_id: str) -> dict[str, Any]:
    return {
        "candidate_item_ids": request.candidate_item_ids,
        "decoding_params": request.decoding_params,
        "metadata": request.metadata,
        "model": request.model,
        "prompt": request.prompt,
        "prompt_version": request.prompt_version,
        "provider": request.provider,
        "sample_id": sample_id,
    }


def _response_usage(response: LLMResponse) -> dict[str, Any]:
    return {
        "cache_hit": response.cache_hit,
        "completion_tokens": response.completion_tokens,
        "latency_ms": response.latency_ms,
        "prompt_tokens": response.prompt_tokens,
        "total_tokens": response.total_tokens,
    }


def _grounding_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        rows.append(
            {
                "evidence_grounding_rate": row.get("evidence_grounding_rate", 0.0),
                "prompt_variant": row.get("prompt_variant"),
                "semantic_evidence_usage_rate": row.get("semantic_evidence_usage_rate", 0.0),
                "time_evidence_usage_rate": row.get("time_evidence_usage_rate", 0.0),
                "transition_evidence_usage_rate": row.get("transition_evidence_usage_rate", 0.0),
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / float(len(values) or 1)
