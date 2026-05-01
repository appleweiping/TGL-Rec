"""Case sampling for Phase 3B API micro diagnostics."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from llm4rec.diagnostics.llm_sequence_time_runner import (
    _candidate_example,
    _candidate_subset,
    _find_example,
    _interactions_by_user,
    _normalize_case_evidence,
    _parse_time_bucket_pairs,
    _prompt_example_from_sample,
    _scores_from_case,
)
from llm4rec.io.artifacts import read_jsonl


DEFAULT_API_CASE_GROUPS = [
    "semantic_and_transition",
    "semantic_only",
    "transition_only",
    "cross_category_transition",
    "high_time_window_strength",
]


def sample_api_micro_cases(
    *,
    source_run_dir: str | Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Sample small, deterministic API diagnostic cases from Phase 2B/3A artifacts."""

    run_dir = Path(source_run_dir)
    diagnostic = dict(config.get("diagnostic", {}))
    groups = [str(group) for group in diagnostic.get("case_groups", DEFAULT_API_CASE_GROUPS)]
    max_cases_per_group = int(diagnostic.get("max_cases_per_group", 5))
    max_total_cases = int(diagnostic.get("max_total_cases", len(groups) * max_cases_per_group))
    candidate_size = int(diagnostic.get("candidate_size", 10))
    random_seed = int(diagnostic.get("random_seed", config.get("experiment", {}).get("seed", 2026)))
    require_target = bool(diagnostic.get("require_target_in_candidates", True))
    min_transition_count = float(diagnostic.get("min_transition_count", 0.0))
    min_time_window_score = float(diagnostic.get("min_time_window_score", 0.0))

    processed_dir = run_dir / "artifacts" / "processed_dataset"
    interactions = read_jsonl(processed_dir / "interactions.jsonl")
    item_records = {str(row["item_id"]): row for row in read_jsonl(processed_dir / "items.jsonl")}
    candidate_rows = [
        row
        for row in read_jsonl(processed_dir / "candidates.jsonl")
        if str(row.get("split")) == "test"
    ]
    grouped_cases = _load_grouped_cases(run_dir)
    grouped_cases["high_time_window_strength"] = _high_time_window_cases(grouped_cases)

    by_user = _interactions_by_user(interactions)
    examples = [_candidate_example(row, by_user) for row in candidate_rows]
    source_to_examples: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        for item_id in example["history"]:
            source_to_examples.setdefault(item_id, []).append(example)

    rng = random.Random(random_seed)
    samples: list[dict[str, Any]] = []
    used: set[tuple[str, str, str, str]] = set()
    for group in groups:
        cases = _eligible_cases(
            grouped_cases.get(group, []),
            min_transition_count=min_transition_count,
            min_time_window_score=min_time_window_score,
        )
        rng.shuffle(cases)
        cases.sort(key=_case_sort_key)
        selected = 0
        for case in cases:
            if selected >= max_cases_per_group or len(samples) >= max_total_cases:
                break
            sample = _case_to_sample(
                group=group,
                case=case,
                examples=examples,
                source_to_examples=source_to_examples,
                item_records=item_records,
                candidate_size=candidate_size,
                require_target_in_candidates=require_target,
            )
            if sample is None:
                continue
            key = (
                str(sample["group"]),
                str(sample["user_id"]),
                str(sample["target_item"]),
                str(sample["evidence_target_item"]),
            )
            if key in used:
                continue
            used.add(key)
            samples.append(sample)
            selected += 1
    return samples


def prompt_example_from_api_sample(sample: dict[str, Any], item_records: dict[str, dict[str, Any]]):
    """Build the shared PromptExample for a sampled API case."""

    return _prompt_example_from_sample(sample, item_records)


def parse_time_bucket_pairs(value: Any) -> dict[tuple[str, str], str]:
    """Parse serialized time-bucket pair keys."""

    return _parse_time_bucket_pairs(value)


def load_item_records(source_run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load item records for prompt building."""

    path = Path(source_run_dir) / "artifacts" / "processed_dataset" / "items.jsonl"
    return {str(row["item_id"]): row for row in read_jsonl(path)}


def load_interaction_catalog(source_run_dir: str | Path) -> set[str]:
    """Load item catalog IDs from the processed Phase 2B dataset."""

    return set(load_item_records(source_run_dir))


def load_transition_edges(source_run_dir: str | Path) -> list[dict[str, Any]]:
    """Load directed transition edges from any known Phase 2B artifact location."""

    root = Path(source_run_dir)
    candidates = [
        root / "diagnostics" / "transition_edges.jsonl",
        root / "artifacts" / "graphs" / "directed_transition_edges.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return read_jsonl(path)
    return []


def load_time_window_edges(source_run_dir: str | Path) -> list[dict[str, Any]]:
    """Load time-window graph edges from Phase 2B artifacts."""

    root = Path(source_run_dir)
    rows: list[dict[str, Any]] = []
    for label in ("1h", "1d", "7d", "30d", "full"):
        for path in (
            root / "diagnostics" / f"time_window_edges_{label}.jsonl",
            root / "artifacts" / "graphs" / f"time_window_edges_{label}.jsonl",
        ):
            if path.is_file():
                rows.extend(read_jsonl(path))
                break
    return rows


def _load_grouped_cases(source_run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = source_run_dir / "diagnostics" / "grouped_cases.json"
    if path.is_file():
        import json

        return dict(json.loads(path.read_text(encoding="utf-8")))
    path = source_run_dir / "similarity_vs_transition.json"
    if path.is_file():
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload.get("grouped_cases", {}))
    return {}


def _high_time_window_cases(grouped_cases: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_cases = [case for cases in grouped_cases.values() for case in cases]
    return sorted(
        all_cases,
        key=lambda row: (
            -_case_time_score(row),
            str(row.get("source_item")),
            str(row.get("target_item")),
        ),
    )


def _eligible_cases(
    cases: list[dict[str, Any]],
    *,
    min_transition_count: float,
    min_time_window_score: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        if float(case.get("transition_count", 0.0) or 0.0) < min_transition_count:
            continue
        if _case_time_score(case) < min_time_window_score:
            continue
        output.append(case)
    return output


def _case_sort_key(case: dict[str, Any]) -> tuple[float, float, str, str]:
    return (
        -float(case.get("transition_score", case.get("transition_count", 0.0)) or 0.0),
        -_case_time_score(case),
        str(case.get("source_item")),
        str(case.get("target_item")),
    )


def _case_time_score(case: dict[str, Any]) -> float:
    return max(
        float(case.get("time_window_score_1d", 0.0) or 0.0),
        float(case.get("time_window_score_7d", 0.0) or 0.0),
        float(case.get("time_window_score_30d", 0.0) or 0.0),
    )


def _case_to_sample(
    *,
    group: str,
    case: dict[str, Any],
    examples: list[dict[str, Any]],
    source_to_examples: dict[str, list[dict[str, Any]]],
    item_records: dict[str, dict[str, Any]],
    candidate_size: int,
    require_target_in_candidates: bool,
) -> dict[str, Any] | None:
    source = str(case.get("source_item"))
    evidence_target = str(case.get("target_item"))
    if source not in item_records or evidence_target not in item_records:
        return None
    candidate_example = _find_example(
        source_to_examples,
        examples,
        source=source,
        target=evidence_target,
    )
    if candidate_example is None:
        return None
    if (
        require_target_in_candidates
        and candidate_example["target_item"] not in candidate_example["candidate_items"]
    ):
        return None
    candidates = _candidate_subset(
        candidate_example["candidate_items"],
        required=[candidate_example["target_item"], evidence_target, source],
        item_records=item_records,
        candidate_size=candidate_size,
    )
    evidence = _normalize_case_evidence(case)
    time_bucket_by_pair = {f"{source}->{evidence_target}": str(case.get("dominant_gap_bucket", ""))}
    return {
        **candidate_example,
        "candidate_items": candidates,
        "case_group": group,
        "contrastive_evidence": [evidence],
        "evidence_source_item": source,
        "evidence_target_item": evidence_target,
        "group": group,
        "sample_id": f"{group}__{candidate_example['user_id']}__{source}__{evidence_target}",
        "time_bucket_by_pair": time_bucket_by_pair,
        "time_window_evidence": [evidence],
        "time_window_scores": _scores_from_case(case, prefix="time_window"),
        "transition_evidence": [evidence],
        "transition_scores": {
            evidence_target: float(
                case.get("transition_score", case.get("transition_count", 0.0)) or 0.0
            )
        },
    }
