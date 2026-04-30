"""History-order perturbation diagnostics for CPU sanity baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tglrec.data import schema
from tglrec.eval.metrics import evaluate_rankings, rank_by_score
from tglrec.models.sanity_baselines import (
    DEFAULT_KS,
    EvaluationCase,
    IncrementalTrainingStats,
    _cases_from_frame,
    _event_precedes,
    _history_only_events,
    _history_splits,
    _segment_row,
    _split_column,
    _validate_interactions,
)
from tglrec.utils.config import write_config
from tglrec.utils.io import ensure_dir, write_json
from tglrec.utils.logging import current_git_commit


DEFAULT_HISTORY_PERTURBATIONS = (
    "original",
    "history_shuffle",
    "order_reversal",
    "timestamp_removal",
    "timestamp_randomization",
    "window_swap",
)
_PERTURBATION_SALTS = {
    "original": 0,
    "history_shuffle": 104729,
    "order_reversal": 130363,
    "timestamp_removal": 155921,
    "timestamp_randomization": 184433,
    "window_swap": 208351,
}
_DAY_SECONDS = 24 * 60 * 60
_WITHIN_WEEK_SECONDS = 7 * _DAY_SECONDS
_LONG_GAP_SECONDS = 30 * _DAY_SECONDS


@dataclass(frozen=True)
class HistoryPerturbationResult:
    """Summary of a completed history perturbation diagnostic run."""

    output_dir: Path
    metrics: dict[str, dict[str, dict[str, float]]]
    deltas: dict[str, dict[str, dict[str, dict[str, float | None]]]]
    num_cases: int


@dataclass(frozen=True)
class HistoryEvent:
    """One history event available before a prediction case."""

    user_id: int
    item_id: int
    timestamp: int | None
    event_id: int


def run_history_perturbation_diagnostics(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    split_name: str = "temporal_leave_one_out",
    eval_split: str = "test",
    ks: tuple[int, ...] = DEFAULT_KS,
    perturbations: tuple[str, ...] = DEFAULT_HISTORY_PERTURBATIONS,
    item_knn_neighbors: int = 50,
    item_knn_max_history_items: int = 100,
    cooccurrence_history_window: int = 200,
    use_validation_history_for_test: bool = True,
    exclude_seen: bool = True,
    seed: int = 2026,
    command: str = "tglrec evaluate history-perturbations",
) -> HistoryPerturbationResult:
    """Evaluate original and perturbed histories for popularity and item-kNN.

    Perturbations alter only the per-case scoring history window. Global
    popularity and co-occurrence statistics remain training-only and
    timestamp-strict, matching the sanity baseline evaluator. Timestamp
    perturbations use only timestamps already present in the user's pre-target
    history, so they cannot introduce future evidence.
    """

    dataset_root = Path(dataset_dir)
    interactions_path = dataset_root / "interactions.csv"
    items_path = dataset_root / "items.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing processed interactions: {interactions_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing processed items: {items_path}")

    selected_perturbations = _validate_perturbations(perturbations)
    interactions = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path)
    split_col = _split_column(split_name)
    _validate_interactions(interactions, split_col)
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    if not ks:
        raise ValueError("ks must contain at least one cutoff")

    item_universe = {int(item_id) for item_id in items[schema.ITEM_ID].tolist()}
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
    rankings: dict[str, dict[str, dict[int, list[int]]]] = {
        baseline: {perturbation: {} for perturbation in selected_perturbations}
        for baseline in ("popularity", "item_knn")
    }
    case_rank_rows: list[dict[str, Any]] = []
    positives: dict[int, int] = {}
    segment_rows: list[dict[str, str]] = []

    stats = IncrementalTrainingStats()
    history_events: dict[int, list[HistoryEvent]] = defaultdict(list)
    train_index = 0
    history_only_index = 0
    for case in sorted(eval_cases, key=lambda row: (row.timestamp, row.user_id, row.event_id)):
        while train_index < len(train_events) and train_events[train_index].timestamp < case.timestamp:
            train_case = train_events[train_index]
            should_append_history = train_case.item_id not in stats.user_items.get(train_case.user_id, set())
            stats.add_event(
                train_case.user_id,
                train_case.item_id,
                train_case.timestamp,
                cooccurrence_history_window=cooccurrence_history_window,
            )
            if should_append_history:
                history_events[train_case.user_id].append(_history_event_from_case(train_case))
            train_index += 1
        while history_only_index < len(history_only_events) and _event_precedes(
            history_only_events[history_only_index], case
        ):
            history_case = history_only_events[history_only_index]
            should_append_history = history_case.item_id not in stats.user_items.get(history_case.user_id, set())
            stats.add_user_history_event(
                history_case.user_id,
                history_case.item_id,
                history_case.timestamp,
            )
            if should_append_history:
                history_events[history_case.user_id].append(_history_event_from_case(history_case))
            history_only_index += 1

        candidates = set(item_universe)
        if exclude_seen:
            candidates.difference_update(stats.user_items.get(case.user_id, set()))
        candidates.add(case.item_id)

        case_key = case.event_id
        positives[case_key] = case.item_id
        base_history_events = _scoring_history_window(
            list(history_events.get(case.user_id, [])),
            max_history_items=item_knn_max_history_items,
        )
        popularity_full_ranking = [
            int(item_id) for item_id in rank_by_score(stats.popularity_scores(candidates))
        ]
        popularity_original_rank = _positive_rank(popularity_full_ranking, case.item_id)
        item_knn_original_rank: int | None = None
        for perturbation in selected_perturbations:
            perturbed_history_events = _perturb_history_events(
                base_history_events,
                perturbation,
                case=case,
                seed=seed,
            )
            perturbed_history = [event.item_id for event in perturbed_history_events]
            rankings["popularity"][perturbation][case_key] = popularity_full_ranking[:max_k]
            if perturbation != "original":
                _append_case_rank_row(
                    case_rank_rows,
                    case=case,
                    baseline="popularity",
                    perturbation=perturbation,
                    original_rank=popularity_original_rank,
                    perturbed_rank=popularity_original_rank,
                    original_history_events=base_history_events,
                    perturbed_history_events=perturbed_history_events,
                    ks=ks,
                )
            item_knn_full_ranking = [
                int(item_id)
                for item_id in rank_by_score(
                    stats.item_knn_scores_for_history(
                        perturbed_history,
                        candidates,
                        neighbors=item_knn_neighbors,
                        max_history_items=0,
                    )
                )
            ]
            perturbed_rank = _positive_rank(item_knn_full_ranking, case.item_id)
            if perturbation == "original":
                item_knn_original_rank = perturbed_rank
            if item_knn_original_rank is None:
                raise RuntimeError("original item-kNN rank must be computed before perturbation ranks")
            rankings["item_knn"][perturbation][case_key] = item_knn_full_ranking[:max_k]
            if perturbation != "original":
                _append_case_rank_row(
                    case_rank_rows,
                    case=case,
                    baseline="item_knn",
                    perturbation=perturbation,
                    original_rank=item_knn_original_rank,
                    perturbed_rank=perturbed_rank,
                    original_history_events=base_history_events,
                    perturbed_history_events=perturbed_history_events,
                    ks=ks,
                )
        segment_rows.append(_segment_row(case, stats))

    metrics = {
        baseline: {
            perturbation: evaluate_rankings(variant_rankings, positives, ks=ks)
            for perturbation, variant_rankings in baseline_rankings.items()
        }
        for baseline, baseline_rankings in rankings.items()
    }
    deltas = _compute_deltas(metrics)
    run_root = Path(output_dir) if output_dir is not None else _default_run_dir()
    _write_run_outputs(
        run_root,
        command=command,
        config={
            "baseline_names": ["popularity", "item_knn"],
            "candidate_mode": "full_ranking",
            "cooccurrence_history_window": cooccurrence_history_window,
            "dataset_dir": str(dataset_root),
            "diagnostic_name": "history_perturbations",
            "eval_split": eval_split,
            "exclude_seen": exclude_seen,
            "history_perturbation_scope": "fixed_scoring_history_events",
            "timestamp_perturbation_semantics": {
                "timestamp_removal": "set history timestamps to null while preserving item order",
                "timestamp_randomization": "permute observed pre-target history timestamps within the fixed scoring window",
                "window_swap": "swap within-week and long-gap observed history timestamps within the fixed scoring window",
            },
            "history_splits": _history_splits(
                eval_split,
                use_validation_history_for_test=use_validation_history_for_test,
            ),
            "item_knn_max_history_items": item_knn_max_history_items,
            "item_knn_neighbors": item_knn_neighbors,
            "ks": list(ks),
            "perturbations": list(selected_perturbations),
            "seed": seed,
            "split_name": split_name,
        },
        metrics=metrics,
        deltas=deltas,
        segment_rows=segment_rows,
        case_rank_rows=case_rank_rows,
        ks=ks,
        rankings=rankings,
        positives=positives,
        num_items=len(item_universe),
        num_eval_users=len({case.user_id for case in eval_cases}),
    )
    return HistoryPerturbationResult(
        output_dir=run_root,
        metrics=metrics,
        deltas=deltas,
        num_cases=len(eval_cases),
    )


def _validate_perturbations(perturbations: tuple[str, ...]) -> tuple[str, ...]:
    if not perturbations:
        raise ValueError("perturbations must contain at least one variant")
    unknown = sorted(set(perturbations) - set(_PERTURBATION_SALTS))
    if unknown:
        raise ValueError(f"Unknown history perturbations: {unknown}")
    ordered: list[str] = []
    for perturbation in ("original", *perturbations):
        if perturbation not in ordered:
            ordered.append(perturbation)
    return tuple(ordered)


def _perturb_history(
    history_items: list[int],
    perturbation: str,
    *,
    case: EvaluationCase,
    seed: int,
) -> list[int]:
    history_events = [
        HistoryEvent(
            user_id=case.user_id,
            item_id=item_id,
            timestamp=None,
            event_id=index,
        )
        for index, item_id in enumerate(history_items)
    ]
    return [
        event.item_id
        for event in _perturb_history_events(history_events, perturbation, case=case, seed=seed)
    ]


def _perturb_history_events(
    history_events: list[HistoryEvent],
    perturbation: str,
    *,
    case: EvaluationCase,
    seed: int,
) -> list[HistoryEvent]:
    if perturbation == "original":
        perturbed = list(history_events)
    elif perturbation == "order_reversal":
        perturbed = list(reversed(history_events))
    elif perturbation == "history_shuffle":
        perturbed = list(history_events)
        rng = random.Random(_case_seed(seed, perturbation, case))
        rng.shuffle(perturbed)
    elif perturbation == "timestamp_removal":
        perturbed = [
            HistoryEvent(
                user_id=event.user_id,
                item_id=event.item_id,
                timestamp=None,
                event_id=event.event_id,
            )
            for event in history_events
        ]
    elif perturbation == "timestamp_randomization":
        timestamps = [event.timestamp for event in history_events]
        shuffled_timestamps = list(timestamps)
        rng = random.Random(_case_seed(seed, perturbation, case))
        rng.shuffle(shuffled_timestamps)
        if len(set(timestamps)) > 1 and shuffled_timestamps == timestamps:
            shuffled_timestamps = shuffled_timestamps[1:] + shuffled_timestamps[:1]
        perturbed = [
            HistoryEvent(
                user_id=event.user_id,
                item_id=event.item_id,
                timestamp=timestamp,
                event_id=event.event_id,
            )
            for event, timestamp in zip(history_events, shuffled_timestamps, strict=True)
        ]
    elif perturbation == "window_swap":
        perturbed = _swap_within_week_and_long_gap_timestamps(history_events, case)
    else:
        raise ValueError(f"Unknown history perturbation: {perturbation}")
    _validate_no_future_timestamps(perturbed, case)
    return perturbed


def _swap_within_week_and_long_gap_timestamps(
    history_events: list[HistoryEvent],
    case: EvaluationCase,
) -> list[HistoryEvent]:
    timestamp_by_index = [event.timestamp for event in history_events]
    within_week_indices: list[int] = []
    long_gap_indices: list[int] = []
    for index, event in enumerate(history_events):
        if event.timestamp is None:
            continue
        gap = case.timestamp - event.timestamp
        if gap <= _WITHIN_WEEK_SECONDS:
            within_week_indices.append(index)
        elif gap > _LONG_GAP_SECONDS:
            long_gap_indices.append(index)

    if not within_week_indices or not long_gap_indices:
        return list(history_events)

    for short_index, long_index in zip(within_week_indices, long_gap_indices, strict=False):
        timestamp_by_index[short_index], timestamp_by_index[long_index] = (
            timestamp_by_index[long_index],
            timestamp_by_index[short_index],
        )

    return [
        HistoryEvent(
            user_id=event.user_id,
            item_id=event.item_id,
            timestamp=timestamp_by_index[index],
            event_id=event.event_id,
        )
        for index, event in enumerate(history_events)
    ]


def _history_event_from_case(case: EvaluationCase) -> HistoryEvent:
    return HistoryEvent(
        user_id=case.user_id,
        item_id=case.item_id,
        timestamp=case.timestamp,
        event_id=case.event_id,
    )


def _validate_no_future_timestamps(history_events: list[HistoryEvent], case: EvaluationCase) -> None:
    for event in history_events:
        if event.timestamp is None:
            continue
        if event.timestamp > case.timestamp:
            raise RuntimeError(
                f"Perturbed history event {event.event_id} has future timestamp "
                f"{event.timestamp} for target case {case.event_id} at {case.timestamp}"
            )
        if (
            event.user_id == case.user_id
            and event.timestamp == case.timestamp
            and event.event_id >= case.event_id
        ):
            raise RuntimeError(
                f"Perturbed history event {event.event_id} does not precede target case "
                f"{case.event_id} at timestamp {case.timestamp}"
            )


def _scoring_history_window(history_events: list[HistoryEvent], *, max_history_items: int) -> list[HistoryEvent]:
    """Return the fixed model-input window before applying perturbations."""

    if max_history_items > 0:
        return list(history_events[-max_history_items:])
    return list(history_events)


def _case_seed(seed: int, perturbation: str, case: EvaluationCase) -> int:
    salt = _PERTURBATION_SALTS[perturbation]
    return (
        int(seed) * 1_000_003
        + int(case.user_id) * 10_007
        + int(case.event_id) * 101
        + int(case.timestamp)
        + salt
    )


def _compute_deltas(
    metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    output: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for baseline, baseline_metrics in metrics.items():
        original = baseline_metrics["original"]
        output[baseline] = {}
        for perturbation, perturbation_metrics in baseline_metrics.items():
            if perturbation == "original":
                continue
            output[baseline][perturbation] = {}
            for metric_name, perturbed_value in sorted(perturbation_metrics.items()):
                original_value = original[metric_name]
                sensitivity_index = None
                if original_value > 0.0:
                    sensitivity_index = (original_value - perturbed_value) / original_value
                output[baseline][perturbation][metric_name] = {
                    "delta_from_original": perturbed_value - original_value,
                    "original": original_value,
                    "perturbed": perturbed_value,
                    "sensitivity_index": sensitivity_index,
                }
    return output


def _positive_rank(ranked_items: list[int], positive_item: int) -> int:
    for rank, item_id in enumerate(ranked_items, start=1):
        if item_id == positive_item:
            return rank
    raise ValueError(f"positive item {positive_item} missing from candidate ranking")


def _append_case_rank_row(
    rows: list[dict[str, Any]],
    *,
    case: EvaluationCase,
    baseline: str,
    perturbation: str,
    original_rank: int,
    perturbed_rank: int,
    original_history_events: list[HistoryEvent],
    perturbed_history_events: list[HistoryEvent],
    ks: tuple[int, ...],
) -> None:
    original_timestamps = [event.timestamp for event in original_history_events]
    perturbed_timestamps = [event.timestamp for event in perturbed_history_events]
    original_items = [event.item_id for event in original_history_events]
    perturbed_items = [event.item_id for event in perturbed_history_events]
    row: dict[str, Any] = {
        "case_id": case.event_id,
        "user_id": case.user_id,
        "target_item_id": case.item_id,
        "target_timestamp": case.timestamp,
        "model": baseline,
        "perturbation": perturbation,
        "history_event_count": len(original_history_events),
        "item_position_changed_count": _changed_position_count(original_items, perturbed_items),
        "timestamp_changed_count": _changed_position_count(original_timestamps, perturbed_timestamps),
        "original_timestamp_null_count": sum(timestamp is None for timestamp in original_timestamps),
        "perturbed_timestamp_null_count": sum(timestamp is None for timestamp in perturbed_timestamps),
        "original_within_week_count": _timestamp_bucket_count(original_timestamps, case, "within_week"),
        "perturbed_within_week_count": _timestamp_bucket_count(perturbed_timestamps, case, "within_week"),
        "original_long_gap_count": _timestamp_bucket_count(original_timestamps, case, "long_gap"),
        "perturbed_long_gap_count": _timestamp_bucket_count(perturbed_timestamps, case, "long_gap"),
        "original_history_fingerprint": _history_fingerprint(original_history_events),
        "perturbed_history_fingerprint": _history_fingerprint(perturbed_history_events),
        "original_rank": original_rank,
        "perturbed_rank": perturbed_rank,
        "rank_delta": perturbed_rank - original_rank,
    }
    for k in sorted(set(ks)):
        original_hit = int(original_rank <= k)
        perturbed_hit = int(perturbed_rank <= k)
        row[f"original_hit@{k}"] = original_hit
        row[f"perturbed_hit@{k}"] = perturbed_hit
        row[f"hit_delta@{k}"] = perturbed_hit - original_hit
    rows.append(row)


def _changed_position_count(original_values: list[Any], perturbed_values: list[Any]) -> int:
    return sum(
        original != perturbed
        for original, perturbed in zip(original_values, perturbed_values, strict=True)
    )


def _timestamp_bucket_count(
    timestamps: list[int | None],
    case: EvaluationCase,
    bucket: str,
) -> int:
    count = 0
    for timestamp in timestamps:
        if timestamp is None:
            continue
        gap = case.timestamp - timestamp
        if bucket == "within_week" and gap <= _WITHIN_WEEK_SECONDS:
            count += 1
        elif bucket == "long_gap" and gap > _LONG_GAP_SECONDS:
            count += 1
    return count


def _history_fingerprint(history_events: list[HistoryEvent]) -> str:
    payload = "|".join(
        f"{event.event_id}:{event.item_id}:{'' if event.timestamp is None else event.timestamp}"
        for event in history_events
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{stamp}-history-perturbations"


def _write_run_outputs(
    output_dir: Path,
    *,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, float]]],
    deltas: dict[str, dict[str, dict[str, dict[str, float | None]]]],
    segment_rows: list[dict[str, str]],
    case_rank_rows: list[dict[str, Any]],
    ks: tuple[int, ...],
    rankings: dict[str, dict[str, dict[int, list[int]]]],
    positives: dict[int, int],
    num_items: int,
    num_eval_users: int,
) -> None:
    root = ensure_dir(output_dir)
    write_config(config, root / "config.yaml")
    write_json(
        {
            "baselines": metrics,
            "candidate_mode": config["candidate_mode"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "deltas": deltas,
            "diagnostic_name": config["diagnostic_name"],
            "num_eval_cases": len(positives),
            "num_eval_users": num_eval_users,
            "num_items": num_items,
        },
        root / "metrics.json",
    )
    _write_metrics_by_perturbation(root / "metrics_by_perturbation.csv", metrics)
    _write_metrics_delta(root / "metrics_delta.csv", deltas)
    _write_metrics_by_segment(root / "metrics_by_segment.csv", metrics, segment_rows, rankings, positives)
    _write_metrics_by_case(root / "metrics_by_case.csv", case_rank_rows, ks=ks)
    _write_environment(root / "environment.json")
    (root / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    (root / "git_commit.txt").write_text(
        current_git_commit(".") + "\n", encoding="utf-8", newline="\n"
    )
    (root / "stdout.log").write_text(
        json.dumps({"output_dir": str(root), "metrics": metrics, "deltas": deltas}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "stderr.log").write_text("", encoding="utf-8", newline="\n")


def _write_metrics_by_perturbation(
    path: Path,
    metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    metric_names = sorted(next(iter(next(iter(metrics.values())).values())).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline", "perturbation", *metric_names],
        )
        writer.writeheader()
        for baseline, baseline_metrics in metrics.items():
            for perturbation, perturbation_metrics in baseline_metrics.items():
                row: dict[str, Any] = {"baseline": baseline, "perturbation": perturbation}
                row.update(perturbation_metrics)
                writer.writerow(row)


def _write_metrics_delta(
    path: Path,
    deltas: dict[str, dict[str, dict[str, dict[str, float | None]]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline",
                "perturbation",
                "metric",
                "original",
                "perturbed",
                "delta_from_original",
                "sensitivity_index",
            ],
        )
        writer.writeheader()
        for baseline, baseline_deltas in deltas.items():
            for perturbation, perturbation_deltas in baseline_deltas.items():
                for metric_name, values in perturbation_deltas.items():
                    writer.writerow(
                        {
                            "baseline": baseline,
                            "perturbation": perturbation,
                            "metric": metric_name,
                            **values,
                        }
                    )


def _write_metrics_by_segment(
    path: Path,
    metrics: dict[str, dict[str, dict[str, float]]],
    segment_rows: list[dict[str, str]],
    rankings: dict[str, dict[str, dict[int, list[int]]]],
    positives: dict[int, int],
) -> None:
    segment_names = [
        "history_length_bucket",
        "last_interaction_gap_bucket",
        "target_popularity_bucket",
        "target_cold_warm",
        "semantic_vs_transition_case_type",
    ]
    metric_names = sorted(next(iter(next(iter(metrics.values())).values())).keys())
    cutoffs = tuple(int(name.split("@")[1]) for name in metric_names if name.startswith("HR@"))
    rows: list[dict[str, Any]] = []
    by_case = {int(row["case_id"]): row for row in segment_rows}
    for baseline, baseline_rankings in rankings.items():
        for perturbation, perturbation_rankings in baseline_rankings.items():
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
                    subset_rankings = {case_id: perturbation_rankings[case_id] for case_id in case_ids}
                    subset_positives = {case_id: positives[case_id] for case_id in case_ids}
                    subset_metrics = evaluate_rankings(
                        subset_rankings,
                        subset_positives,
                        ks=cutoffs,
                    )
                    output_row: dict[str, Any] = {
                        "baseline": baseline,
                        "perturbation": perturbation,
                        "segment_name": segment_name,
                        "segment_value": segment_value,
                        "num_cases": len(case_ids),
                    }
                    output_row.update(subset_metrics)
                    rows.append(output_row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline",
                "perturbation",
                "segment_name",
                "segment_value",
                "num_cases",
                *metric_names,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_by_case(path: Path, rows: list[dict[str, Any]], *, ks: tuple[int, ...]) -> None:
    hit_fields = [
        field
        for cutoff in sorted(set(ks))
        for field in (
            f"original_hit@{cutoff}",
            f"perturbed_hit@{cutoff}",
            f"hit_delta@{cutoff}",
        )
    ]
    fieldnames = [
        "case_id",
        "user_id",
        "target_item_id",
        "target_timestamp",
        "model",
        "perturbation",
        "history_event_count",
        "item_position_changed_count",
        "timestamp_changed_count",
        "original_timestamp_null_count",
        "perturbed_timestamp_null_count",
        "original_within_week_count",
        "perturbed_within_week_count",
        "original_long_gap_count",
        "perturbed_long_gap_count",
        "original_history_fingerprint",
        "perturbed_history_fingerprint",
        "original_rank",
        "perturbed_rank",
        "rank_delta",
        *hit_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda value: (
                str(value["model"]),
                str(value["perturbation"]),
                int(value["target_timestamp"]),
                int(value["user_id"]),
                int(value["case_id"]),
            ),
        ):
            writer.writerow(row)


def _write_environment(path: Path) -> None:
    write_json(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        path,
    )
