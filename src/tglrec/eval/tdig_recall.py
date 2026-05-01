"""As-of TDIG direct-transition candidate recall evaluation."""

from __future__ import annotations

import csv
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.metadata
import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import CHECKSUM_MANIFEST_NAME, file_fingerprint, write_checksum_manifest
from tglrec.eval.metrics import rank_by_score
from tglrec.graph.tdig import GAP_BUCKETS, gap_bucket
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

DEFAULT_TDIG_RECALL_SCORE_FIELD = "transition_probability"

SEMANTIC_VS_TRANSITION_CASE_TYPES = (
    "semantic_and_transition",
    "semantic_only",
    "transition_only",
    "neither_semantic_nor_transition",
    "not_computed",
)
_ITEM_ID_COLUMNS = {schema.ITEM_ID, schema.RAW_ITEM_ID}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_METADATA_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "item",
    "movie",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class TDIGRecallResult:
    """Summary of a completed TDIG candidate recall run."""

    output_dir: Path
    metrics: dict[str, float]
    num_cases: int


@dataclass(frozen=True)
class _ScoredCandidate:
    item_id: int
    score: float
    support: int
    last_transition_timestamp: int
    source_item_ids: tuple[int, ...]


@dataclass(frozen=True)
class _SemanticTransitionLabel:
    case_type: str
    semantic_overlap_max: int | None
    semantic_overlap_source_item_id: int | None
    semantic_overlap_tokens: tuple[str, ...]
    target_has_transition_evidence: bool


class _IncrementalTDIG:
    """Train-only directed transition counts available before a prediction time."""

    def __init__(self, *, include_same_timestamp_transitions: bool = False) -> None:
        self.include_same_timestamp_transitions = include_same_timestamp_transitions
        self.previous_train_event: dict[int, EvaluationCase | None] = {}
        self.ambiguous_previous_timestamp: dict[int, int] = {}
        self.edge_observations: dict[tuple[int, int], dict[str, Any]] = {}
        self.outgoing_targets: dict[int, set[int]] = defaultdict(set)
        self.source_counts: Counter[int] = Counter()
        self.target_counts: Counter[int] = Counter()
        self.item_popularity: Counter[int] = Counter()
        self.total_transitions = 0
        self.skipped_same_timestamp_tie_groups = 0
        self.skipped_same_timestamp_adjacent_transitions = 0
        self.skipped_same_timestamp_ambiguous_bridges = 0

    def add_train_event(self, event: EvaluationCase) -> None:
        self.item_popularity[event.item_id] += 1
        ambiguous_timestamp = self.ambiguous_previous_timestamp.get(event.user_id)
        if ambiguous_timestamp is not None:
            if event.timestamp == ambiguous_timestamp:
                self.skipped_same_timestamp_adjacent_transitions += 1
                self.previous_train_event[event.user_id] = None
                return
            del self.ambiguous_previous_timestamp[event.user_id]
            self.previous_train_event[event.user_id] = event
            self.skipped_same_timestamp_ambiguous_bridges += 1
            return
        previous = self.previous_train_event.get(event.user_id)
        if previous is not None:
            if (
                not self.include_same_timestamp_transitions
                and event.timestamp == previous.timestamp
            ):
                self.skipped_same_timestamp_tie_groups += 1
                self.skipped_same_timestamp_adjacent_transitions += 1
                self.previous_train_event[event.user_id] = None
                self.ambiguous_previous_timestamp[event.user_id] = event.timestamp
                return
            source_item_id = previous.item_id
            target_item_id = event.item_id
            gap_seconds = max(0, event.timestamp - previous.timestamp)
            key = (source_item_id, target_item_id)
            observation = self.edge_observations.setdefault(
                key,
                {
                    "support": 0,
                    "timestamps": [],
                    "gaps": [],
                    "gap_histogram": Counter(),
                },
            )
            observation["support"] += 1
            observation["timestamps"].append(event.timestamp)
            observation["gaps"].append(gap_seconds)
            observation["gap_histogram"][gap_bucket(gap_seconds)] += 1
            self.outgoing_targets[source_item_id].add(target_item_id)
            self.source_counts[source_item_id] += 1
            self.target_counts[target_item_id] += 1
            self.total_transitions += 1
        self.previous_train_event[event.user_id] = event

    def retrieve_from_sources(
        self,
        source_item_ids: list[int],
        *,
        per_source_top_k: int,
        score_field: str,
        aggregation: str,
        gap_bucket_name: str | None,
    ) -> list[_ScoredCandidate]:
        if per_source_top_k <= 0:
            raise ValueError(f"per_source_top_k must be positive, got {per_source_top_k}")
        if aggregation not in {"max", "sum"}:
            raise ValueError("aggregation must be 'max' or 'sum'")
        if gap_bucket_name is not None and gap_bucket_name not in GAP_BUCKETS:
            raise ValueError(f"Unknown TDIG gap bucket: {gap_bucket_name}")

        combined: dict[int, dict[str, Any]] = {}
        for source_item_id in source_item_ids:
            scored_edges = [
                (
                    target_item_id,
                    self._score_observation(
                        source_item_id,
                        target_item_id,
                        self.edge_observations[(source_item_id, target_item_id)],
                        score_field=score_field,
                        gap_bucket_name=gap_bucket_name,
                    ),
                )
                for target_item_id in self.outgoing_targets.get(source_item_id, set())
            ]
            scored_edges = [
                (target_item_id, score)
                for target_item_id, score in scored_edges
                if score > 0.0
            ]
            for target_item_id, score in sorted(
                scored_edges,
                key=lambda pair: (
                    -pair[1],
                    -int(self.edge_observations[(source_item_id, pair[0])]["support"]),
                    -int(max(self.edge_observations[(source_item_id, pair[0])]["timestamps"])),
                    pair[0],
                ),
            )[:per_source_top_k]:
                observation = self.edge_observations[(source_item_id, target_item_id)]
                candidate = combined.setdefault(
                    target_item_id,
                    {
                        "score": 0.0 if aggregation == "sum" else None,
                        "support": 0,
                        "last_transition_timestamp": 0,
                        "source_item_ids": set(),
                    },
                )
                if aggregation == "sum":
                    candidate["score"] += score
                else:
                    candidate["score"] = max(float(candidate["score"] or 0.0), score)
                candidate["support"] += int(observation["support"])
                candidate["last_transition_timestamp"] = max(
                    int(candidate["last_transition_timestamp"]),
                    int(max(observation["timestamps"])),
                )
                candidate["source_item_ids"].add(source_item_id)

        return [
            _ScoredCandidate(
                item_id=item_id,
                score=float(values["score"] or 0.0),
                support=int(values["support"]),
                last_transition_timestamp=int(values["last_transition_timestamp"]),
                source_item_ids=tuple(sorted(int(item) for item in values["source_item_ids"])),
            )
            for item_id, values in combined.items()
        ]

    def _score_observation(
        self,
        source_item_id: int,
        target_item_id: int,
        observation: dict[str, Any],
        *,
        score_field: str,
        gap_bucket_name: str | None,
    ) -> float:
        if gap_bucket_name is not None:
            return float(observation["gap_histogram"].get(gap_bucket_name, 0))
        support = int(observation["support"])
        if score_field == "support":
            return float(support)
        transition_probability = support / float(self.source_counts[source_item_id])
        if score_field == "transition_probability":
            return transition_probability
        target_probability = (
            self.target_counts[target_item_id] / self.total_transitions
            if self.total_transitions
            else 0.0
        )
        lift = transition_probability / target_probability if target_probability > 0 else 0.0
        if score_field == "lift":
            return lift
        if score_field == "pmi":
            return math.log(lift) if lift > 0.0 else 0.0
        raise ValueError(f"Unknown edge score field: {score_field}")


def run_tdig_candidate_recall(
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
    seed: int = 2026,
    command: str = "tglrec evaluate tdig-candidate-recall",
) -> TDIGRecallResult:
    """Evaluate as-of direct-transition TDIG candidate recall."""

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

    interactions = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path)
    split_col = _split_column(split_name)
    _validate_interactions(interactions, split_col)

    item_universe = {int(item_id) for item_id in items[schema.ITEM_ID].tolist()}
    item_tokens = _build_item_token_index(items)
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

    max_k = max(ks)
    positives: dict[int, int] = {}
    rankings: dict[int, list[int]] = {}
    case_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, str]] = []

    graph = _IncrementalTDIG(
        include_same_timestamp_transitions=include_same_timestamp_transitions
    )
    query_history: dict[int, list[EvaluationCase]] = defaultdict(list)
    train_index = 0
    history_only_index = 0
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

        source_items = _recent_unique_items(
            query_history.get(case.user_id, []),
            max_items=source_history_items,
        )
        candidates = graph.retrieve_from_sources(
            source_items,
            per_source_top_k=per_source_top_k,
            score_field=score_field,
            aggregation=aggregation,
            gap_bucket_name=gap_bucket_name,
        )
        seen_items = {event.item_id for event in query_history.get(case.user_id, [])}
        scored_candidates = {
            candidate.item_id: candidate.score
            for candidate in candidates
            if candidate.item_id in item_universe
            and (not exclude_seen or candidate.item_id == case.item_id or candidate.item_id not in seen_items)
        }
        ranked_items = [int(item_id) for item_id in rank_by_score(scored_candidates)]
        top_ranked_items = ranked_items[:max_k]
        case_key = case.event_id
        positives[case_key] = case.item_id
        rankings[case_key] = top_ranked_items
        candidate_by_id = {candidate.item_id: candidate for candidate in candidates}
        target_rank = _positive_rank(ranked_items, case.item_id)
        semantic_transition_label = _label_semantic_vs_transition(
            case,
            source_item_ids=source_items,
            item_tokens=item_tokens,
            target_rank=target_rank,
        )
        case_rows.append(
            _case_row(
                case,
                source_items=source_items,
                ranked_items=ranked_items,
                target_rank=target_rank,
                target_candidate=candidate_by_id.get(case.item_id),
                semantic_transition_label=semantic_transition_label,
                ks=ks,
            )
        )
        segment_rows.append(
            _segment_row(
                case,
                graph,
                query_history.get(case.user_id, []),
                semantic_transition_label=semantic_transition_label,
            )
        )

    metrics = _candidate_recall_metrics(rankings, positives, ks=ks)
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
            "candidate_mode": "tdig_direct_transition_recall",
            "dataset_dir": str(dataset_root),
            "dataset_provenance": _dataset_provenance(dataset_root),
            "eval_split": eval_split,
            "exclude_seen": exclude_seen,
            "gap_bucket": gap_bucket_name,
            "history_splits": _history_splits(
                eval_split,
                use_validation_history_for_test=use_validation_history_for_test,
            ),
            "include_same_timestamp_transitions": include_same_timestamp_transitions,
            "ks": list(ks),
            "leakage_policy": (
                "TDIG transition evidence is updated only from split=train events with "
                "timestamp strictly before each prediction timestamp; optional validation "
                "source-history events must also have timestamp strictly before the test target. "
                "Semantic-vs-transition labels use only processed items.csv metadata and the "
                "same as-of TDIG target ranking used for candidate recall."
            ),
            "per_source_top_k": per_source_top_k,
            "score_field": score_field,
            "seed": seed,
            "same_timestamp_skip_metric_definitions": {
                "same_timestamp_adjacent_transition_skip_count": (
                    "Adjacent same-user train-event pairs skipped because their timestamps "
                    "are identical and include_same_timestamp_transitions is false."
                ),
                "same_timestamp_ambiguous_bridge_skip_count": (
                    "Later chronological bridges skipped after an unresolved same-timestamp "
                    "tie so tied events cannot create downstream transition evidence."
                ),
                "same_timestamp_tie_group_skip_count": (
                    "Same-user identical-timestamp tie groups first encountered by the "
                    "incremental TDIG updater; this is a group count, not an adjacent-pair count."
                ),
            },
            "semantic_vs_transition_labeling": {
                "case_types": list(SEMANTIC_VS_TRANSITION_CASE_TYPES),
                "semantic_signal": (
                    "At least one normalized item metadata token overlaps between the target "
                    "item and the recent source-history items used for TDIG retrieval."
                ),
                "transition_signal": (
                    "The target item is retrieved and ranked by direct TDIG evidence available "
                    "strictly before the prediction timestamp."
                ),
                "token_columns": "All non-id columns in processed items.csv.",
            },
            "source_history_items": source_history_items,
            "split_name": split_name,
        },
        metrics=metrics,
        case_rows=case_rows,
        segment_rows=segment_rows,
        rankings=rankings,
        positives=positives,
        num_eval_users=len({case.user_id for case in eval_cases}),
        num_items=len(item_universe),
        graph=graph,
    )
    return TDIGRecallResult(output_dir=run_root, metrics=metrics, num_cases=len(eval_cases))


def _build_item_token_index(items: pd.DataFrame) -> dict[int, frozenset[str]]:
    """Build deterministic item metadata token sets from processed items.csv columns."""

    text_columns = [column for column in items.columns if column not in _ITEM_ID_COLUMNS]
    token_index: dict[int, frozenset[str]] = {}
    for row in items.to_dict("records"):
        item_id = int(row[schema.ITEM_ID])
        tokens: set[str] = set()
        for column in text_columns:
            tokens.update(_metadata_tokens(row.get(column)))
        token_index[item_id] = frozenset(tokens)
    return token_index


def _metadata_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        output: set[str] = set()
        for entry in value:
            output.update(_metadata_tokens(entry))
        return output
    if isinstance(value, dict):
        output = set()
        for key, entry in value.items():
            output.update(_metadata_tokens(key))
            output.update(_metadata_tokens(entry))
        return output
    text = str(value).lower()
    return {
        token
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 2
        and token not in _METADATA_STOPWORDS
        and any(character.isalpha() for character in token)
    }


def _label_semantic_vs_transition(
    case: EvaluationCase,
    *,
    source_item_ids: list[int],
    item_tokens: dict[int, frozenset[str]],
    target_rank: int | None,
) -> _SemanticTransitionLabel:
    target_has_transition_evidence = target_rank is not None
    target_tokens = item_tokens.get(case.item_id, frozenset())
    if not source_item_ids:
        return _SemanticTransitionLabel(
            case_type="not_computed",
            semantic_overlap_max=None,
            semantic_overlap_source_item_id=None,
            semantic_overlap_tokens=(),
            target_has_transition_evidence=target_has_transition_evidence,
        )
    if not target_tokens:
        return _SemanticTransitionLabel(
            case_type=(
                "transition_only"
                if target_has_transition_evidence
                else "neither_semantic_nor_transition"
            ),
            semantic_overlap_max=0,
            semantic_overlap_source_item_id=source_item_ids[0],
            semantic_overlap_tokens=(),
            target_has_transition_evidence=target_has_transition_evidence,
        )

    best_source_item_id: int | None = None
    best_tokens: tuple[str, ...] = ()
    best_overlap = -1
    has_source_tokens = False
    for source_item_id in source_item_ids:
        source_tokens = item_tokens.get(source_item_id, frozenset())
        if source_tokens:
            has_source_tokens = True
        overlap_tokens = tuple(sorted(target_tokens & source_tokens))
        overlap = len(overlap_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_source_item_id = source_item_id
            best_tokens = overlap_tokens

    if best_source_item_id is None or not has_source_tokens:
        return _SemanticTransitionLabel(
            case_type="not_computed",
            semantic_overlap_max=None,
            semantic_overlap_source_item_id=None,
            semantic_overlap_tokens=(),
            target_has_transition_evidence=target_has_transition_evidence,
        )

    has_semantic_evidence = best_overlap > 0
    if has_semantic_evidence and target_has_transition_evidence:
        case_type = "semantic_and_transition"
    elif has_semantic_evidence:
        case_type = "semantic_only"
    elif target_has_transition_evidence:
        case_type = "transition_only"
    else:
        case_type = "neither_semantic_nor_transition"
    return _SemanticTransitionLabel(
        case_type=case_type,
        semantic_overlap_max=best_overlap,
        semantic_overlap_source_item_id=best_source_item_id,
        semantic_overlap_tokens=best_tokens,
        target_has_transition_evidence=target_has_transition_evidence,
    )


def _recent_unique_items(history: list[EvaluationCase], *, max_items: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for event in reversed(history):
        if event.item_id in seen:
            continue
        seen.add(event.item_id)
        output.append(event.item_id)
        if max_items > 0 and len(output) >= max_items:
            break
    return output


def _candidate_recall_metrics(
    rankings: dict[int, list[int]],
    positives: dict[int, int],
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        hits = [
            float(positives[case_id] in ranked_items[:k])
            for case_id, ranked_items in rankings.items()
        ]
        metrics[f"candidate_recall@{k}"] = sum(hits) / len(hits)
    return metrics


def _positive_rank(ranked_items: list[int], positive_item: int) -> int | None:
    for rank, item_id in enumerate(ranked_items, start=1):
        if item_id == positive_item:
            return rank
    return None


def _same_timestamp_skip_metrics(graph: _IncrementalTDIG) -> dict[str, float]:
    return {
        "same_timestamp_adjacent_transition_skip_count": float(
            graph.skipped_same_timestamp_adjacent_transitions
        ),
        "same_timestamp_ambiguous_bridge_skip_count": float(
            graph.skipped_same_timestamp_ambiguous_bridges
        ),
        "same_timestamp_tie_group_skip_count": float(graph.skipped_same_timestamp_tie_groups),
    }


def _case_row(
    case: EvaluationCase,
    *,
    source_items: list[int],
    ranked_items: list[int],
    target_rank: int | None,
    target_candidate: _ScoredCandidate | None,
    semantic_transition_label: _SemanticTransitionLabel,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.event_id,
        "user_id": case.user_id,
        "target_item_id": case.item_id,
        "target_timestamp": case.timestamp,
        "source_history_item_count": len(source_items),
        "source_item_ids_json": json.dumps(source_items),
        "target_rank": "" if target_rank is None else target_rank,
        "target_score": "" if target_candidate is None else target_candidate.score,
        "target_support": "" if target_candidate is None else target_candidate.support,
        "target_source_item_ids_json": (
            "[]" if target_candidate is None else json.dumps(list(target_candidate.source_item_ids))
        ),
        "target_has_transition_evidence": int(
            semantic_transition_label.target_has_transition_evidence
        ),
        "semantic_overlap_max": (
            ""
            if semantic_transition_label.semantic_overlap_max is None
            else semantic_transition_label.semantic_overlap_max
        ),
        "semantic_overlap_source_item_id": (
            ""
            if semantic_transition_label.semantic_overlap_source_item_id is None
            else semantic_transition_label.semantic_overlap_source_item_id
        ),
        "semantic_overlap_tokens_json": json.dumps(
            list(semantic_transition_label.semantic_overlap_tokens)
        ),
        "semantic_vs_transition_case_type": semantic_transition_label.case_type,
        "top_candidate_ids_json": json.dumps(ranked_items[: max(ks)]),
    }
    for k in sorted(set(ks)):
        row[f"candidate_recall@{k}"] = int(target_rank is not None and target_rank <= k)
    return row


def _segment_row(
    case: EvaluationCase,
    graph: _IncrementalTDIG,
    history: list[EvaluationCase],
    *,
    semantic_transition_label: _SemanticTransitionLabel,
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
        "semantic_vs_transition_case_type": semantic_transition_label.case_type,
    }


def _history_bucket(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 4:
        return "1-4"
    if length <= 19:
        return "5-19"
    if length <= 99:
        return "20-99"
    return "100+"


def _gap_bucket(gap_seconds: int | None) -> str:
    if gap_seconds is None:
        return "unknown"
    day = 24 * 60 * 60
    if gap_seconds <= day:
        return "<=1d"
    if gap_seconds <= 7 * day:
        return "<=7d"
    if gap_seconds <= 30 * day:
        return "<=30d"
    return ">30d"


def _popularity_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 19:
        return "5-19"
    if count <= 99:
        return "20-99"
    return "100+"


def _default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{stamp}-tdig-candidate-recall"


def _write_run_outputs(
    output_dir: Path,
    *,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    case_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, str]],
    rankings: dict[int, list[int]],
    positives: dict[int, int],
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
            "evaluator": "tdig_direct_candidate_recall",
            "metrics": metrics,
            "num_eval_cases": len(positives),
            "num_eval_users": num_eval_users,
            "num_items": num_items,
            "num_tdig_edges_observed": len(graph.edge_observations),
            "num_tdig_transitions_observed": graph.total_transitions,
            "same_timestamp_skip_counts": _same_timestamp_skip_metrics(graph),
        },
        root / "metrics.json",
    )
    _write_metrics_by_case(root / "metrics_by_case.csv", case_rows, ks=tuple(config["ks"]))
    _write_metrics_by_segment(
        root / "metrics_by_segment.csv",
        metrics,
        segment_rows,
        rankings,
        positives,
    )
    _write_environment(root / "environment.json")
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
    hit_fields = [f"candidate_recall@{k}" for k in sorted(set(ks))]
    fieldnames = [
        "case_id",
        "user_id",
        "target_item_id",
        "target_timestamp",
        "source_history_item_count",
        "source_item_ids_json",
        "target_rank",
        "target_score",
        "target_support",
        "target_source_item_ids_json",
        "target_has_transition_evidence",
        "semantic_overlap_max",
        "semantic_overlap_source_item_id",
        "semantic_overlap_tokens_json",
        "semantic_vs_transition_case_type",
        "top_candidate_ids_json",
        *hit_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (int(value["target_timestamp"]), int(value["case_id"]))):
            writer.writerow(row)


def _write_metrics_by_segment(
    path: Path,
    metrics: dict[str, float],
    segment_rows: list[dict[str, str]],
    rankings: dict[int, list[int]],
    positives: dict[int, int],
) -> None:
    segment_names = [
        "history_length_bucket",
        "last_interaction_gap_bucket",
        "target_popularity_bucket",
        "target_cold_warm",
        "semantic_vs_transition_case_type",
    ]
    metric_names = sorted(name for name in metrics if name.startswith("candidate_recall@"))
    cutoffs = tuple(int(name.rsplit("@", 1)[1]) for name in metric_names if "@" in name)
    rows: list[dict[str, Any]] = []
    by_case = {int(row["case_id"]): row for row in segment_rows}
    for segment_name in segment_names:
        values = sorted({row[segment_name] for row in segment_rows})
        for segment_value in values:
            case_ids = [
                case_id
                for case_id, row in by_case.items()
                if row[segment_name] == segment_value and case_id in positives
            ]
            if not case_ids:
                continue
            subset_rankings = {case_id: rankings[case_id] for case_id in case_ids}
            subset_positives = {case_id: positives[case_id] for case_id in case_ids}
            subset_metrics = _candidate_recall_metrics(
                subset_rankings,
                subset_positives,
                ks=cutoffs,
            )
            output_row: dict[str, Any] = {
                "segment_name": segment_name,
                "segment_value": segment_value,
                "num_cases": len(case_ids),
            }
            output_row.update(subset_metrics)
            rows.append(output_row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["segment_name", "segment_value", "num_cases", *metric_names],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_environment(path: Path) -> None:
    write_json(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "package_versions": _package_versions(["tglrec", "pandas", "numpy", "pyyaml"]),
        },
        path,
    )


def _dataset_provenance(dataset_root: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for name in ("interactions.csv", "items.csv", "config.yaml", "metadata.json", CHECKSUM_MANIFEST_NAME):
        path = dataset_root / name
        if path.is_file():
            provenance[name] = file_fingerprint(path, include_path=True)
        else:
            provenance[name] = {"missing": True, "path": str(path)}
    return provenance


def _current_git_status() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - depends on local Git availability.
        return f"UNAVAILABLE: {exc}\n"
    output = result.stdout
    if result.stderr:
        output += ("\n" if output and not output.endswith("\n") else "") + result.stderr
    if result.returncode != 0:
        output += f"\nUNAVAILABLE: git status exited {result.returncode}\n"
    return output if output.endswith("\n") else output + "\n"


def _package_versions(package_names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions
