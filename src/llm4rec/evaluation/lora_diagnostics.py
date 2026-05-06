"""Diagnostics for local LoRA reranking outputs."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import ensure_dir, iter_jsonl, write_csv_rows, write_json


def diagnose_lora_predictions(predictions_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Summarize output-shape and candidate-protocol issues for LoRA predictions."""

    predictions = Path(predictions_path)
    out_dir = ensure_dir(output_dir)
    rows = list(iter_jsonl(predictions))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "")), str(row.get("method", "")))].append(row)

    summary_rows = [
        _summarize_subset(dataset, method, subset)
        for (dataset, method), subset in sorted(grouped.items())
    ]
    overall = _summarize_subset("ALL", "ALL", rows)
    write_csv_rows(out_dir / "lora_prediction_diagnostics.csv", summary_rows)
    payload = {"overall": overall, "by_dataset_method": summary_rows}
    write_json(out_dir / "lora_prediction_diagnostics.json", payload)
    return payload


def _summarize_subset(dataset: str, method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"dataset": dataset, "method": method, "num_rows": 0}
    parse_success = 0
    raw_nonempty = 0
    prompt_continuations = 0
    json_like = 0
    bare_id_only = 0
    recovered_target_top1 = 0
    target_in_candidates = 0
    target_at_last_candidate = 0
    target_at_top10 = 0
    candidate_sizes: list[int] = []
    target_candidate_ranks: list[int] = []
    for row in rows:
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("parse_success", False):
            parse_success += 1
        raw_output = str(row.get("raw_output") or "").strip()
        if raw_output:
            raw_nonempty += 1
        if _looks_like_prompt_continuation(raw_output):
            prompt_continuations += 1
        if raw_output.startswith("{") or '"ranked_item_ids"' in raw_output:
            json_like += 1
        if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9]*", raw_output):
            bare_id_only += 1
        target = str(row.get("target_item", ""))
        candidates = [str(item) for item in row.get("candidate_items", [])]
        predicted = [str(item) for item in row.get("predicted_items", [])]
        candidate_sizes.append(len(candidates))
        if target in candidates:
            target_in_candidates += 1
            rank = candidates.index(target) + 1
            target_candidate_ranks.append(rank)
            if rank == len(candidates):
                target_at_last_candidate += 1
        if predicted and predicted[0] == target:
            recovered_target_top1 += 1
        if target in predicted[:10]:
            target_at_top10 += 1
    return {
        "bare_id_only_rate": bare_id_only / total,
        "dataset": dataset,
        "json_like_rate": json_like / total,
        "max_candidate_size": max(candidate_sizes),
        "mean_candidate_size": sum(candidate_sizes) / total,
        "mean_target_candidate_rank": _mean(target_candidate_ranks),
        "method": method,
        "min_candidate_size": min(candidate_sizes),
        "num_rows": total,
        "parse_success_rate": parse_success / total,
        "prompt_continuation_rate": prompt_continuations / total,
        "raw_nonempty_rate": raw_nonempty / total,
        "target_at_last_candidate_rate": target_at_last_candidate / total,
        "target_in_candidates_rate": target_in_candidates / total,
        "target_predicted_top10_rate": target_at_top10 / total,
        "target_predicted_top1_rate": recovered_target_top1 / total,
    }


def _looks_like_prompt_continuation(raw_output: str) -> bool:
    return (
        "\nuser:" in raw_output
        or "\nassistant:" in raw_output
        or "Candidates:" in raw_output
        or "History:" in raw_output
    )


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
