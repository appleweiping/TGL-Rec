"""Semantic-vs-transition hard-candidate stress diagnostics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import write_checksum_manifest
from tglrec.eval.metrics import rank_by_score
from tglrec.eval.tdig_recall import (
    DEFAULT_TDIG_RECALL_SCORE_FIELD,
    SEMANTIC_VS_TRANSITION_CASE_TYPES,
    _IncrementalTDIG,
    _build_item_token_index,
    _current_git_status,
    _dataset_provenance,
    _gap_bucket,
    _history_bucket,
    _package_versions,
    _popularity_bucket,
    _recent_unique_items,
    _same_timestamp_skip_metrics,
)
from tglrec.graph.tdig import GAP_BUCKETS
from tglrec.models.sanity_baselines import (
    DEFAULT_KS,
    EvaluationCase,
    _cases_from_frame,
    _history_only_events,
    _history_splits,
    _split_column,
    _validate_interactions,
)
from tglrec.utils.config import write_config
from tglrec.utils.io import ensure_dir, write_json
from tglrec.utils.logging import current_git_commit

STRESS_RANKERS = ("semantic_overlap", "tdig_transition", "popularity")
STRESS_CASE_TYPES = (
    "target_transition_with_semantic_negative",
    "target_transition_without_semantic_negative",
    "semantic_negative_without_target_transition",
    "neither_stress_signal",
)


@dataclass(frozen=True)
class SemanticTransitionStressResult:
    """Summary of a completed hard-candidate stress run."""

    output_dir: Path
    metrics: dict[str, float]
    num_cases: int


@dataclass(frozen=True)
class _SemanticEvidence:
    overlap: int
    source_item_id: int | None
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class _CandidateEvidence:
    item_id: int
    semantic_overlap: int
    semantic_source_item_id: int | None
    semantic_tokens: tuple[str, ...]
    transition_score: float
    transition_support: int
    transition_source_item_ids: tuple[int, ...]
    popularity: int


@dataclass(frozen=True)
class _SemanticNeighbor:
    item_id: int
    source_item_id: int
    overlap: int
    tokens: tuple[str, ...]


def run_semantic_transition_stress(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    split_name: str = "temporal_leave_one_out",
    eval_split: str = "test",
    ks: tuple[int, ...] = DEFAULT_KS,
    source_history_items: int = 20,
    max_history_items: int | None = None,
    per_source_top_k: int = 50,
    score_field: str = DEFAULT_TDIG_RECALL_SCORE_FIELD,
    aggregation: str = "max",
    gap_bucket_name: str | None = None,
    use_validation_history_for_test: bool = True,
    exclude_seen: bool = True,
    include_same_timestamp_transitions: bool = False,
    max_eval_cases: int | None = None,
    seed: int = 2026,
    command: str = "tglrec evaluate semantic-transition-stress",
) -> SemanticTransitionStressResult:
    """Build hard candidate sets and score them with simple diagnostic rankers."""

    dataset_root = Path(dataset_dir)
    interactions_path = dataset_root / "interactions.csv"
    items_path = dataset_root / "items.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing processed interactions: {interactions_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing processed items: {items_path}")
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    if not ks:
        raise ValueError("ks must contain at least one cutoff")
    if any(k <= 0 for k in ks):
        raise ValueError(f"ks must contain only positive cutoffs, got {ks}")
    if max_history_items is not None:
        source_history_items = max_history_items
    if source_history_items < 0:
        raise ValueError("source_history_items must be non-negative")
    if per_source_top_k <= 0:
        raise ValueError("per_source_top_k must be positive")
    if max_eval_cases is not None and max_eval_cases <= 0:
        raise ValueError("max_eval_cases must be positive when provided")
    if aggregation not in {"max", "sum"}:
        raise ValueError("aggregation must be 'max' or 'sum'")
    if gap_bucket_name is not None and gap_bucket_name not in GAP_BUCKETS:
        raise ValueError(f"Unknown TDIG gap bucket: {gap_bucket_name}")

    interactions = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path)
    split_col = _split_column(split_name)
    _validate_interactions(interactions, split_col)

    item_universe = {int(item_id) for item_id in items[schema.ITEM_ID].tolist()}
    item_tokens = _build_item_token_index(items)
    token_to_items = _build_token_to_items(item_tokens)
    interaction_items = {int(item_id) for item_id in interactions[schema.ITEM_ID].tolist()}
    missing_items = sorted(interaction_items - item_universe)
    if missing_items:
        preview = ", ".join(str(item_id) for item_id in missing_items[:5])
        raise ValueError(
            f"items.csv is missing {len(missing_items)} interaction item ids, e.g. {preview}"
        )

    train_events = _cases_from_frame(interactions.loc[interactions[split_col] == "train"])
    history_only_events = _history_only_events(
        interactions,
        split_col=split_col,
        eval_split=eval_split,
        use_validation_history_for_test=use_validation_history_for_test,
    )
    eval_cases = _cases_from_frame(interactions.loc[interactions[split_col] == eval_split])
    if not eval_cases:
        raise ValueError(f"No {eval_split} cases found in {interactions_path}")
    if max_eval_cases is not None:
        eval_cases = eval_cases[:max_eval_cases]

    graph = _IncrementalTDIG(
        include_same_timestamp_transitions=include_same_timestamp_transitions
    )
    query_history: dict[int, list[EvaluationCase]] = defaultdict(list)
    semantic_neighbor_cache: dict[int, list[_SemanticNeighbor]] = {}
    train_index = 0
    history_only_index = 0
    case_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, str]] = []

    sorted_eval_cases = sorted(eval_cases, key=lambda row: (row.timestamp, row.user_id, row.event_id))
    for case in sorted_eval_cases:
        while train_index < len(train_events) and train_events[train_index].timestamp < case.timestamp:
            train_event = train_events[train_index]
            graph.add_train_event(train_event)
            query_history[train_event.user_id].append(train_event)
            train_index += 1
        while (
            history_only_index < len(history_only_events)
            and history_only_events[history_only_index].timestamp < case.timestamp
        ):
            history_event = history_only_events[history_only_index]
            query_history[history_event.user_id].append(history_event)
            history_only_index += 1

        history = query_history.get(case.user_id, [])
        source_items = _recent_unique_items(history, max_items=source_history_items)
        tdig_candidates = graph.retrieve_from_sources(
            source_items,
            per_source_top_k=per_source_top_k,
            score_field=score_field,
            aggregation=aggregation,
            gap_bucket_name=gap_bucket_name,
        )
        seen_items = {event.item_id for event in history}
        transition_by_id = {
            candidate.item_id: candidate
            for candidate in tdig_candidates
            if candidate.item_id in item_universe
            and (not exclude_seen or candidate.item_id == case.item_id or candidate.item_id not in seen_items)
        }
        eligible_negative_items = _eligible_negative_items(
            item_universe,
            target_item_id=case.item_id,
            seen_items=seen_items,
            exclude_seen=exclude_seen,
        )
        target = _candidate_evidence(
            case.item_id,
            source_item_ids=source_items,
            item_tokens=item_tokens,
            transition_by_id=transition_by_id,
            item_popularity=graph.item_popularity,
        )
        semantic_negative = _select_semantic_hard_negative(
            eligible_negative_items,
            source_item_ids=source_items,
            item_tokens=item_tokens,
            token_to_items=token_to_items,
            semantic_neighbor_cache=semantic_neighbor_cache,
            transition_by_id=transition_by_id,
            item_popularity=graph.item_popularity,
        )
        selected_negative_ids = {
            candidate.item_id
            for candidate in (semantic_negative,)
            if candidate is not None
        }
        transition_negative = _select_transition_hard_negative(
            eligible_negative_items - selected_negative_ids,
            source_item_ids=source_items,
            item_tokens=item_tokens,
            transition_by_id=transition_by_id,
            item_popularity=graph.item_popularity,
        )
        if transition_negative is not None:
            selected_negative_ids.add(transition_negative.item_id)
        popularity_negative = _select_popularity_hard_negative(
            eligible_negative_items - selected_negative_ids,
            source_item_ids=source_items,
            item_tokens=item_tokens,
            transition_by_id=transition_by_id,
            item_popularity=graph.item_popularity,
        )
        if popularity_negative is not None:
            selected_negative_ids.add(popularity_negative.item_id)
        random_negative = _select_random_negative(
            eligible_negative_items - selected_negative_ids,
            case=case,
            seed=seed,
            source_item_ids=source_items,
            item_tokens=item_tokens,
            transition_by_id=transition_by_id,
            item_popularity=graph.item_popularity,
        )

        evidence_by_item = _candidate_set_by_item(
            target=target,
            semantic_negative=semantic_negative,
            transition_negative=transition_negative,
            popularity_negative=popularity_negative,
            random_negative=random_negative,
        )
        ranked_by_ranker = {
            ranker: _rank_candidates(evidence_by_item, ranker=ranker)
            for ranker in STRESS_RANKERS
        }
        case_row = _case_row(
            case,
            source_items=source_items,
            target=target,
            semantic_negative=semantic_negative,
            transition_negative=transition_negative,
            popularity_negative=popularity_negative,
            random_negative=random_negative,
            evidence_by_item=evidence_by_item,
            ranked_by_ranker=ranked_by_ranker,
            ks=ks,
        )
        case_rows.append(case_row)
        segment_rows.append(_segment_row(case, graph, history, case_row=case_row))

    metrics = _aggregate_case_metrics(case_rows)
    metrics.update(
        {
            "num_tdig_edges_as_of_final_case": float(len(graph.edge_observations)),
            "num_tdig_transitions_as_of_final_case": float(graph.total_transitions),
            **_same_timestamp_skip_metrics(graph),
        }
    )
    run_root = Path(output_dir) if output_dir is not None else _default_run_dir()
    _write_run_outputs(
        run_root,
        command=command,
        config={
            "aggregation": aggregation,
            "candidate_mode": "semantic_transition_hard_candidates",
            "dataset_dir": str(dataset_root),
            "dataset_provenance": _dataset_provenance(dataset_root),
            "diagnostic_rankers": list(STRESS_RANKERS),
            "eval_split": eval_split,
            "exclude_seen": exclude_seen,
            "gap_bucket": gap_bucket_name,
            "hard_candidate_roles": [
                "target",
                "semantic_hard_negative",
                "transition_hard_negative",
                "popularity_hard_negative",
                "random_negative",
            ],
            "history_splits": _history_splits(
                eval_split,
                use_validation_history_for_test=use_validation_history_for_test,
            ),
            "include_same_timestamp_transitions": include_same_timestamp_transitions,
            "ks": list(ks),
            "leakage_policy": (
                "TDIG transition evidence and item popularity are updated only from split=train "
                "events with timestamp strictly before each prediction timestamp. Optional "
                "validation source-history events are used only if their timestamp is strictly "
                "before the test target. Semantic hard negatives use processed items.csv metadata "
                "and do not inspect future interactions."
            ),
            "per_source_top_k": per_source_top_k,
            "score_field": score_field,
            "seed": seed,
            "max_eval_cases": max_eval_cases,
            "semantic_vs_transition_case_types": list(SEMANTIC_VS_TRANSITION_CASE_TYPES),
            "source_history_items": source_history_items,
            "split_name": split_name,
            "stress_case_types": list(STRESS_CASE_TYPES),
        },
        metrics=metrics,
        case_rows=case_rows,
        segment_rows=segment_rows,
        num_eval_users=len({case.user_id for case in eval_cases}),
        num_items=len(item_universe),
        graph=graph,
    )
    return SemanticTransitionStressResult(
        output_dir=run_root,
        metrics=metrics,
        num_cases=len(eval_cases),
    )


def _eligible_negative_items(
    item_universe: set[int],
    *,
    target_item_id: int,
    seen_items: set[int],
    exclude_seen: bool,
) -> set[int]:
    eligible = set(item_universe)
    eligible.discard(target_item_id)
    if exclude_seen:
        eligible.difference_update(seen_items)
    return eligible


def _candidate_evidence(
    item_id: int,
    *,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
    transition_by_id: dict[int, Any],
    item_popularity: dict[int, int],
    semantic_evidence: _SemanticEvidence | None = None,
) -> _CandidateEvidence:
    semantic = semantic_evidence or _semantic_evidence(item_id, source_item_ids, item_tokens)
    transition = transition_by_id.get(item_id)
    return _CandidateEvidence(
        item_id=int(item_id),
        semantic_overlap=semantic.overlap,
        semantic_source_item_id=semantic.source_item_id,
        semantic_tokens=semantic.tokens,
        transition_score=0.0 if transition is None else float(transition.score),
        transition_support=0 if transition is None else int(transition.support),
        transition_source_item_ids=(
            () if transition is None else tuple(int(item) for item in transition.source_item_ids)
        ),
        popularity=int(item_popularity.get(item_id, 0)),
    )


def _build_token_to_items(
    item_tokens: dict[int, frozenset[str]],
) -> dict[str, frozenset[int]]:
    token_to_items: dict[str, set[int]] = defaultdict(set)
    for item_id, tokens in item_tokens.items():
        for token in tokens:
            token_to_items[token].add(item_id)
    return {
        token: frozenset(sorted(matching_item_ids))
        for token, matching_item_ids in token_to_items.items()
    }


def _semantic_evidence(
    item_id: int,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
) -> _SemanticEvidence:
    target_tokens = item_tokens.get(item_id, frozenset())
    if not target_tokens or not source_item_ids:
        return _SemanticEvidence(overlap=0, source_item_id=None, tokens=())
    best_source_item_id: int | None = None
    best_tokens: tuple[str, ...] = ()
    best_overlap = 0
    for source_item_id in source_item_ids:
        source_tokens = item_tokens.get(source_item_id, frozenset())
        overlap_tokens = tuple(sorted(target_tokens & source_tokens))
        if len(overlap_tokens) > best_overlap:
            best_overlap = len(overlap_tokens)
            best_source_item_id = source_item_id
            best_tokens = overlap_tokens
    return _SemanticEvidence(
        overlap=best_overlap,
        source_item_id=best_source_item_id,
        tokens=best_tokens,
    )


def _select_semantic_hard_negative(
    eligible_items: set[int],
    *,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
    token_to_items: dict[str, frozenset[int]],
    semantic_neighbor_cache: dict[int, list[_SemanticNeighbor]],
    transition_by_id: dict[int, Any],
    item_popularity: dict[int, int],
) -> _CandidateEvidence | None:
    best: _CandidateEvidence | None = None
    best_key: tuple[float, bool, float, int, int] | None = None
    for source_item_id in source_item_ids:
        neighbors = _semantic_neighbors_for_source(
            source_item_id,
            item_tokens=item_tokens,
            token_to_items=token_to_items,
            semantic_neighbor_cache=semantic_neighbor_cache,
        )
        for neighbor in neighbors:
            if best is not None and neighbor.overlap < best.semantic_overlap:
                break
            if neighbor.item_id not in eligible_items:
                continue
            candidate = _candidate_evidence(
                neighbor.item_id,
                source_item_ids=source_item_ids,
                item_tokens=item_tokens,
                transition_by_id=transition_by_id,
                item_popularity=item_popularity,
                semantic_evidence=_SemanticEvidence(
                    overlap=neighbor.overlap,
                    source_item_id=neighbor.source_item_id,
                    tokens=neighbor.tokens,
                ),
            )
            candidate_key = (
                -float(candidate.semantic_overlap),
                candidate.transition_score > 0.0,
                candidate.transition_score,
                -candidate.popularity,
                candidate.item_id,
            )
            if best_key is None or candidate_key < best_key:
                best = candidate
                best_key = candidate_key
    return best


def _semantic_neighbors_for_source(
    source_item_id: int,
    *,
    item_tokens: dict[int, frozenset[str]],
    token_to_items: dict[str, frozenset[int]],
    semantic_neighbor_cache: dict[int, list[_SemanticNeighbor]],
) -> list[_SemanticNeighbor]:
    cached = semantic_neighbor_cache.get(source_item_id)
    if cached is not None:
        return cached
    source_tokens = item_tokens.get(source_item_id, frozenset())
    token_hits_by_item: dict[int, list[str]] = defaultdict(list)
    for token in sorted(source_tokens):
        for item_id in sorted(token_to_items.get(token, frozenset())):
            token_hits_by_item[item_id].append(token)
    neighbors = sorted(
        (
            _SemanticNeighbor(
                item_id=item_id,
                source_item_id=source_item_id,
                overlap=len(tokens),
                tokens=tuple(sorted(tokens)),
            )
            for item_id, tokens in token_hits_by_item.items()
            if tokens
        ),
        key=lambda neighbor: (
            -neighbor.overlap,
            neighbor.item_id,
        ),
    )
    semantic_neighbor_cache[source_item_id] = neighbors
    return neighbors


def _select_transition_hard_negative(
    eligible_items: set[int],
    *,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
    transition_by_id: dict[int, Any],
    item_popularity: dict[int, int],
) -> _CandidateEvidence | None:
    candidates = [
        _candidate_evidence(
            item_id,
            source_item_ids=source_item_ids,
            item_tokens=item_tokens,
            transition_by_id=transition_by_id,
            item_popularity=item_popularity,
        )
        for item_id in eligible_items
        if item_id in transition_by_id and transition_by_id[item_id].score > 0.0
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.transition_score,
            candidate.semantic_overlap,
            -candidate.transition_support,
            candidate.item_id,
        ),
    )[0]


def _select_popularity_hard_negative(
    eligible_items: set[int],
    *,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
    transition_by_id: dict[int, Any],
    item_popularity: dict[int, int],
) -> _CandidateEvidence | None:
    if not eligible_items:
        return None
    item_id = min(
        eligible_items,
        key=lambda candidate_id: (-int(item_popularity.get(candidate_id, 0)), candidate_id),
    )
    return _candidate_evidence(
        item_id,
        source_item_ids=source_item_ids,
        item_tokens=item_tokens,
        transition_by_id=transition_by_id,
        item_popularity=item_popularity,
    )


def _select_random_negative(
    eligible_items: set[int],
    *,
    case: EvaluationCase,
    seed: int,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
    transition_by_id: dict[int, Any],
    item_popularity: dict[int, int],
) -> _CandidateEvidence | None:
    if not eligible_items:
        return None
    item_id = min(
        eligible_items,
        key=lambda candidate_id: _stable_random_key(seed, case.event_id, candidate_id),
    )
    return _candidate_evidence(
        item_id,
        source_item_ids=source_item_ids,
        item_tokens=item_tokens,
        transition_by_id=transition_by_id,
        item_popularity=item_popularity,
    )


def _stable_random_key(seed: int, case_id: int, item_id: int) -> int:
    value = (
        (int(seed) + 0x9E3779B97F4A7C15)
        ^ (int(case_id) * 0xBF58476D1CE4E5B9)
        ^ (int(item_id) * 0x94D049BB133111EB)
    ) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _candidate_set_by_item(
    *,
    target: _CandidateEvidence,
    semantic_negative: _CandidateEvidence | None,
    transition_negative: _CandidateEvidence | None,
    popularity_negative: _CandidateEvidence | None,
    random_negative: _CandidateEvidence | None,
) -> dict[int, _CandidateEvidence]:
    output = {target.item_id: target}
    for candidate in (
        semantic_negative,
        transition_negative,
        popularity_negative,
        random_negative,
    ):
        if candidate is not None:
            output.setdefault(candidate.item_id, candidate)
    return output


def _rank_candidates(
    evidence_by_item: dict[int, _CandidateEvidence],
    *,
    ranker: str,
) -> list[int]:
    if ranker == "semantic_overlap":
        scores = {
            item_id: float(evidence.semantic_overlap)
            for item_id, evidence in evidence_by_item.items()
        }
    elif ranker == "tdig_transition":
        scores = {
            item_id: float(evidence.transition_score)
            for item_id, evidence in evidence_by_item.items()
        }
    elif ranker == "popularity":
        scores = {
            item_id: float(evidence.popularity)
            for item_id, evidence in evidence_by_item.items()
        }
    else:
        raise ValueError(f"Unknown stress ranker: {ranker}")
    return [int(item_id) for item_id in rank_by_score(scores)]


def _rank_position(ranked_items: list[int], item_id: int | None) -> int | None:
    if item_id is None:
        return None
    for rank, ranked_item_id in enumerate(ranked_items, start=1):
        if ranked_item_id == item_id:
            return rank
    return None


def _case_row(
    case: EvaluationCase,
    *,
    source_items: list[int],
    target: _CandidateEvidence,
    semantic_negative: _CandidateEvidence | None,
    transition_negative: _CandidateEvidence | None,
    popularity_negative: _CandidateEvidence | None,
    random_negative: _CandidateEvidence | None,
    evidence_by_item: dict[int, _CandidateEvidence],
    ranked_by_ranker: dict[str, list[int]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    semantic_vs_transition_case_type = _semantic_vs_transition_case_type(target)
    stress_case_type = _stress_case_type(target, semantic_negative)
    row: dict[str, Any] = {
        "case_id": case.event_id,
        "user_id": case.user_id,
        "target_item_id": case.item_id,
        "target_timestamp": case.timestamp,
        "source_history_item_count": len(source_items),
        "source_item_ids_json": json.dumps(source_items),
        "candidate_item_ids_json": json.dumps(sorted(evidence_by_item)),
        "target_semantic_overlap": target.semantic_overlap,
        "target_semantic_source_item_id": _optional_int(target.semantic_source_item_id),
        "target_semantic_tokens_json": json.dumps(list(target.semantic_tokens)),
        "target_transition_score": target.transition_score,
        "target_transition_support": target.transition_support,
        "target_transition_source_item_ids_json": json.dumps(
            list(target.transition_source_item_ids)
        ),
        "semantic_negative_item_id": _optional_candidate_item(semantic_negative),
        "semantic_negative_overlap": _optional_candidate_value(
            semantic_negative, "semantic_overlap"
        ),
        "semantic_negative_source_item_id": _optional_candidate_value(
            semantic_negative, "semantic_source_item_id"
        ),
        "semantic_negative_tokens_json": _optional_candidate_tokens(semantic_negative),
        "semantic_negative_transition_score": _optional_candidate_value(
            semantic_negative, "transition_score"
        ),
        "transition_negative_item_id": _optional_candidate_item(transition_negative),
        "transition_negative_score": _optional_candidate_value(
            transition_negative, "transition_score"
        ),
        "transition_negative_support": _optional_candidate_value(
            transition_negative, "transition_support"
        ),
        "transition_negative_semantic_overlap": _optional_candidate_value(
            transition_negative, "semantic_overlap"
        ),
        "popularity_negative_item_id": _optional_candidate_item(popularity_negative),
        "popularity_negative_popularity": _optional_candidate_value(
            popularity_negative, "popularity"
        ),
        "random_negative_item_id": _optional_candidate_item(random_negative),
        "target_has_transition_evidence": int(target.transition_score > 0.0),
        "has_semantic_hard_negative": int(semantic_negative is not None),
        "has_transition_hard_negative": int(transition_negative is not None),
        "semantic_vs_transition_case_type": semantic_vs_transition_case_type,
        "stress_case_type": stress_case_type,
    }
    for ranker in STRESS_RANKERS:
        ranked_items = ranked_by_ranker[ranker]
        target_rank = _rank_position(ranked_items, target.item_id)
        semantic_rank = _rank_position(
            ranked_items,
            None if semantic_negative is None else semantic_negative.item_id,
        )
        transition_negative_rank = _rank_position(
            ranked_items,
            None if transition_negative is None else transition_negative.item_id,
        )
        row[f"{ranker}_ranking_json"] = json.dumps(ranked_items)
        row[f"{ranker}_target_rank"] = target_rank
        row[f"{ranker}_semantic_negative_rank"] = _optional_int(semantic_rank)
        row[f"{ranker}_transition_negative_rank"] = _optional_int(transition_negative_rank)
        row[f"{ranker}_semantic_trap_eligible"] = int(semantic_rank is not None)
        row[f"{ranker}_semantic_trap"] = int(
            semantic_rank is not None and target_rank is not None and semantic_rank < target_rank
        )
        row[f"{ranker}_transition_win_eligible"] = int(
            target.transition_score > 0.0 and semantic_rank is not None
        )
        row[f"{ranker}_transition_win"] = int(
            target.transition_score > 0.0
            and semantic_rank is not None
            and target_rank is not None
            and target_rank < semantic_rank
        )
        row[f"{ranker}_transition_negative_win_eligible"] = int(
            semantic_rank is not None and transition_negative_rank is not None
        )
        row[f"{ranker}_transition_negative_win"] = int(
            semantic_rank is not None
            and transition_negative_rank is not None
            and transition_negative_rank < semantic_rank
        )
        for k in sorted(set(ks)):
            row[f"{ranker}_target_hr@{k}"] = int(target_rank is not None and target_rank <= k)
    return row


def _semantic_vs_transition_case_type(target: _CandidateEvidence) -> str:
    has_semantic = target.semantic_overlap > 0
    has_transition = target.transition_score > 0.0
    if has_semantic and has_transition:
        return "semantic_and_transition"
    if has_semantic:
        return "semantic_only"
    if has_transition:
        return "transition_only"
    return "neither_semantic_nor_transition"


def _stress_case_type(
    target: _CandidateEvidence,
    semantic_negative: _CandidateEvidence | None,
) -> str:
    has_transition = target.transition_score > 0.0
    has_semantic_negative = semantic_negative is not None
    if has_transition and has_semantic_negative:
        return "target_transition_with_semantic_negative"
    if has_transition:
        return "target_transition_without_semantic_negative"
    if has_semantic_negative:
        return "semantic_negative_without_target_transition"
    return "neither_stress_signal"


def _optional_int(value: int | None) -> str | int:
    return "" if value is None else value


def _optional_candidate_item(candidate: _CandidateEvidence | None) -> str | int:
    return "" if candidate is None else candidate.item_id


def _optional_candidate_value(candidate: _CandidateEvidence | None, field: str) -> Any:
    return "" if candidate is None else getattr(candidate, field)


def _optional_candidate_tokens(candidate: _CandidateEvidence | None) -> str:
    return "[]" if candidate is None else json.dumps(list(candidate.semantic_tokens))


def _aggregate_case_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate empty stress rows.")
    metrics: dict[str, float] = {
        "num_cases": float(len(rows)),
        "semantic_hard_negative_coverage": _mean(rows, "has_semantic_hard_negative"),
        "target_transition_evidence_rate": _mean(rows, "target_has_transition_evidence"),
        "transition_hard_negative_coverage": _mean(rows, "has_transition_hard_negative"),
    }
    metrics["stress_case_coverage"] = _rate(
        sum(
            int(row["has_semantic_hard_negative"])
            for row in rows
            if int(row["target_has_transition_evidence"])
        ),
        sum(int(row["target_has_transition_evidence"]) for row in rows),
    )
    for ranker in STRESS_RANKERS:
        target_ranks = [int(row[f"{ranker}_target_rank"]) for row in rows]
        metrics[f"{ranker}_target_top1_rate"] = sum(
            1.0 for rank in target_ranks if rank == 1
        ) / len(target_ranks)
        metrics[f"{ranker}_target_mrr"] = sum(1.0 / rank for rank in target_ranks) / len(
            target_ranks
        )
        metrics[f"{ranker}_semantic_trap_rate"] = _eligible_rate(
            rows,
            numerator_field=f"{ranker}_semantic_trap",
            denominator_field=f"{ranker}_semantic_trap_eligible",
        )
        metrics[f"{ranker}_transition_win_rate"] = _eligible_rate(
            rows,
            numerator_field=f"{ranker}_transition_win",
            denominator_field=f"{ranker}_transition_win_eligible",
        )
        metrics[f"{ranker}_transition_negative_win_rate"] = _eligible_rate(
            rows,
            numerator_field=f"{ranker}_transition_negative_win",
            denominator_field=f"{ranker}_transition_negative_win_eligible",
        )
    return metrics


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _eligible_rate(
    rows: list[dict[str, Any]],
    *,
    numerator_field: str,
    denominator_field: str,
) -> float:
    denominator = sum(int(row[denominator_field]) for row in rows)
    numerator = sum(int(row[numerator_field]) for row in rows)
    return _rate(numerator, denominator)


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _segment_row(
    case: EvaluationCase,
    graph: _IncrementalTDIG,
    history: list[EvaluationCase],
    *,
    case_row: dict[str, Any],
) -> dict[str, str]:
    target_popularity = graph.item_popularity.get(case.item_id, 0)
    last_timestamp = max((event.timestamp for event in history), default=None)
    last_gap = None if last_timestamp is None else max(0, case.timestamp - last_timestamp)
    return {
        "case_id": str(case.event_id),
        "user_id": str(case.user_id),
        "positive_item_id": str(case.item_id),
        "history_length_bucket": _history_bucket(len({event.item_id for event in history})),
        "last_interaction_gap_bucket": _gap_bucket(last_gap),
        "target_popularity_bucket": _popularity_bucket(target_popularity),
        "target_cold_warm": "cold" if target_popularity == 0 else "warm",
        "semantic_vs_transition_case_type": str(case_row["semantic_vs_transition_case_type"]),
        "stress_case_type": str(case_row["stress_case_type"]),
    }


def _default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{stamp}-semantic-transition-stress"


def _write_run_outputs(
    output_dir: Path,
    *,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    case_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, str]],
    num_eval_users: int,
    num_items: int,
    graph: _IncrementalTDIG,
) -> None:
    root = ensure_dir(output_dir)
    write_config(config, root / "config.yaml")
    write_json(
        {
            "candidate_mode": config["candidate_mode"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostic_rankers": list(STRESS_RANKERS),
            "evaluator": "semantic_transition_stress",
            "metrics": metrics,
            "num_eval_cases": len(case_rows),
            "num_eval_users": num_eval_users,
            "num_items": num_items,
            "num_tdig_edges_observed": len(graph.edge_observations),
            "num_tdig_transitions_observed": graph.total_transitions,
            "same_timestamp_skip_counts": _same_timestamp_skip_metrics(graph),
        },
        root / "metrics.json",
    )
    _write_metrics_by_case(root / "metrics_by_case.csv", case_rows, ks=tuple(config["ks"]))
    _write_metrics_by_segment(root / "metrics_by_segment.csv", case_rows, segment_rows)
    write_json(
        {
            "platform": __import__("platform").platform(),
            "python": __import__("sys").version,
            "python_executable": __import__("sys").executable,
            "package_versions": _package_versions(["tglrec", "pandas", "numpy", "pyyaml"]),
        },
        root / "environment.json",
    )
    write_json(
        {
            "completed_successfully": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "This records evaluator completion, not external shell wrapper status.",
        },
        root / "run_status.json",
    )
    (root / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    (root / "git_commit.txt").write_text(
        current_git_commit(".") + "\n", encoding="utf-8", newline="\n"
    )
    (root / "git_status.txt").write_text(_current_git_status(), encoding="utf-8", newline="\n")
    (root / "stdout.log").write_text(
        json.dumps({"output_dir": str(root), "metrics": metrics}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "stderr.log").write_text("", encoding="utf-8", newline="\n")
    write_checksum_manifest(
        root,
        [
            "config.yaml",
            "metrics.json",
            "metrics_by_case.csv",
            "metrics_by_segment.csv",
            "command.txt",
            "git_commit.txt",
            "git_status.txt",
            "run_status.json",
            "stdout.log",
            "stderr.log",
            "environment.json",
        ],
    )


def _write_metrics_by_case(path: Path, rows: list[dict[str, Any]], *, ks: tuple[int, ...]) -> None:
    base_fields = [
        "case_id",
        "user_id",
        "target_item_id",
        "target_timestamp",
        "source_history_item_count",
        "source_item_ids_json",
        "candidate_item_ids_json",
        "target_semantic_overlap",
        "target_semantic_source_item_id",
        "target_semantic_tokens_json",
        "target_transition_score",
        "target_transition_support",
        "target_transition_source_item_ids_json",
        "semantic_negative_item_id",
        "semantic_negative_overlap",
        "semantic_negative_source_item_id",
        "semantic_negative_tokens_json",
        "semantic_negative_transition_score",
        "transition_negative_item_id",
        "transition_negative_score",
        "transition_negative_support",
        "transition_negative_semantic_overlap",
        "popularity_negative_item_id",
        "popularity_negative_popularity",
        "random_negative_item_id",
        "target_has_transition_evidence",
        "has_semantic_hard_negative",
        "has_transition_hard_negative",
        "semantic_vs_transition_case_type",
        "stress_case_type",
    ]
    ranker_fields: list[str] = []
    for ranker in STRESS_RANKERS:
        ranker_fields.extend(
            [
                f"{ranker}_ranking_json",
                f"{ranker}_target_rank",
                f"{ranker}_semantic_negative_rank",
                f"{ranker}_transition_negative_rank",
                f"{ranker}_semantic_trap_eligible",
                f"{ranker}_semantic_trap",
                f"{ranker}_transition_win_eligible",
                f"{ranker}_transition_win",
                f"{ranker}_transition_negative_win_eligible",
                f"{ranker}_transition_negative_win",
            ]
        )
        ranker_fields.extend(f"{ranker}_target_hr@{k}" for k in sorted(set(ks)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base_fields, *ranker_fields])
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (int(value["target_timestamp"]), int(value["case_id"]))):
            writer.writerow(row)


def _write_metrics_by_segment(
    path: Path,
    case_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, str]],
) -> None:
    segment_names = [
        "history_length_bucket",
        "last_interaction_gap_bucket",
        "target_popularity_bucket",
        "target_cold_warm",
        "semantic_vs_transition_case_type",
        "stress_case_type",
    ]
    case_by_id = {int(row["case_id"]): row for row in case_rows}
    segment_by_id = {int(row["case_id"]): row for row in segment_rows}
    rows: list[dict[str, Any]] = []
    metric_names = [
        "semantic_hard_negative_coverage",
        "target_transition_evidence_rate",
        "transition_hard_negative_coverage",
        "stress_case_coverage",
    ]
    for ranker in STRESS_RANKERS:
        metric_names.extend(
            [
                f"{ranker}_semantic_trap_rate",
                f"{ranker}_transition_win_rate",
                f"{ranker}_target_top1_rate",
                f"{ranker}_target_mrr",
            ]
        )
    for segment_name in segment_names:
        values = sorted({row[segment_name] for row in segment_rows})
        for segment_value in values:
            ids = [
                case_id
                for case_id, row in segment_by_id.items()
                if row[segment_name] == segment_value and case_id in case_by_id
            ]
            subset_rows = [case_by_id[case_id] for case_id in ids]
            if not subset_rows:
                continue
            subset_metrics = _aggregate_case_metrics(subset_rows)
            output_row: dict[str, Any] = {
                "segment_name": segment_name,
                "segment_value": segment_value,
                "num_cases": len(subset_rows),
            }
            for metric_name in metric_names:
                output_row[metric_name] = subset_metrics[metric_name]
            rows.append(output_row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["segment_name", "segment_value", "num_cases", *metric_names],
        )
        writer.writeheader()
        writer.writerows(rows)
