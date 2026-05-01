"""Local CPU sanity baselines for sequential recommendation.

The baselines in this module are intentionally simple but not synthetic: global
popularity and co-occurrence statistics are built incrementally from split=train
events available strictly before each prediction timestamp. For final test
evaluation, the user's validation event can be used as prior history without
adding it to global training statistics.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tglrec.data import schema
from tglrec.eval.metrics import evaluate_rankings, rank_by_score
from tglrec.utils.config import write_config
from tglrec.utils.io import ensure_dir, write_json
from tglrec.utils.logging import current_git_commit


DEFAULT_KS = (5, 10, 20)


def _item_id_tie_key(item_id: int) -> tuple[int, int | str]:
    if isinstance(item_id, int):
        return (0, item_id)
    return (1, str(item_id))


@dataclass(frozen=True)
class EvaluationCase:
    """One held-out next-item prediction case."""

    user_id: int
    item_id: int
    timestamp: int
    event_id: int


@dataclass(frozen=True)
class SanityBaselineResult:
    """Summary of a completed sanity baseline run."""

    output_dir: Path
    metrics: dict[str, Any]
    num_cases: int


class IncrementalTrainingStats:
    """Training-only popularity and item co-occurrence state as of a timestamp."""

    def __init__(self) -> None:
        self.item_popularity: dict[int, int] = defaultdict(int)
        self.user_items: dict[int, set[int]] = defaultdict(set)
        self.user_history_items: dict[int, list[int]] = defaultdict(list)
        self.user_last_timestamp: dict[int, int] = {}
        self.cooccurrence: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._cooccurrence_versions: dict[int, int] = defaultdict(int)
        self._top_neighbor_cache: dict[tuple[int, int], tuple[int, list[tuple[int, float]]]] = {}

    def add_event(
        self,
        user_id: int,
        item_id: int,
        timestamp: int,
        *,
        cooccurrence_history_window: int = 200,
    ) -> None:
        """Add one train event in chronological order."""

        self.item_popularity[item_id] += 1
        seen = self.user_items[user_id]
        if item_id not in seen:
            history = self.user_history_items[user_id]
            changed_items = {item_id}
            previous_items = (
                history
                if cooccurrence_history_window <= 0
                else history[-cooccurrence_history_window:]
            )
            for previous_item in previous_items:
                self.cooccurrence[item_id][previous_item] += 1
                self.cooccurrence[previous_item][item_id] += 1
                changed_items.add(previous_item)
            for changed_item in changed_items:
                self._cooccurrence_versions[changed_item] += 1
            seen.add(item_id)
            history.append(item_id)
        previous_timestamp = self.user_last_timestamp.get(user_id)
        if previous_timestamp is None or timestamp > previous_timestamp:
            self.user_last_timestamp[user_id] = timestamp

    def add_user_history_event(self, user_id: int, item_id: int, timestamp: int) -> None:
        """Add a user-observed event without updating global item statistics."""

        seen = self.user_items[user_id]
        if item_id not in seen:
            seen.add(item_id)
            self.user_history_items[user_id].append(item_id)
        previous_timestamp = self.user_last_timestamp.get(user_id)
        if previous_timestamp is None or timestamp > previous_timestamp:
            self.user_last_timestamp[user_id] = timestamp

    def popularity_scores(self, candidates: set[int]) -> dict[int, float]:
        """Score candidates by as-of training popularity."""

        return {item_id: float(self.item_popularity.get(item_id, 0)) for item_id in candidates}

    def item_knn_scores(
        self,
        user_id: int,
        candidates: set[int],
        *,
        neighbors: int,
        max_history_items: int = 100,
    ) -> dict[int, float]:
        """Score candidates by top-k item co-occurrence counts with user history."""

        return self.item_knn_scores_for_history(
            self.user_history_items.get(user_id, []),
            candidates,
            neighbors=neighbors,
            max_history_items=max_history_items,
        )

    def item_knn_scores_for_history(
        self,
        history_items: list[int],
        candidates: set[int],
        *,
        neighbors: int,
        max_history_items: int = 100,
    ) -> dict[int, float]:
        """Score candidates by top-k item co-occurrence counts with an explicit history."""

        if neighbors <= 0:
            raise ValueError(f"neighbors must be positive, got {neighbors}")
        history = history_items
        if max_history_items > 0:
            history = history[-max_history_items:]
        scores = {item_id: 0.0 for item_id in candidates}
        for history_item in history:
            for candidate_item, co_count in self._top_neighbors(history_item, neighbors):
                if candidate_item not in candidates:
                    continue
                scores[candidate_item] += co_count
        return scores

    def _top_neighbors(self, item_id: int, neighbors: int) -> list[tuple[int, float]]:
        cache_key = (item_id, neighbors)
        version = self._cooccurrence_versions.get(item_id, 0)
        cached = self._top_neighbor_cache.get(cache_key)
        if cached is not None and cached[0] == version:
            return cached[1]
        top_neighbors = [
            (neighbor_id, float(co_count))
            for neighbor_id, co_count in sorted(
                self.cooccurrence.get(item_id, {}).items(),
                key=lambda pair: (-pair[1], _item_id_tie_key(pair[0])),
            )[:neighbors]
        ]
        self._top_neighbor_cache[cache_key] = (version, top_neighbors)
        return top_neighbors

    def history_length(self, user_id: int) -> int:
        return len(self.user_items.get(user_id, set()))

    def last_gap_seconds(self, user_id: int, timestamp: int) -> int | None:
        previous = self.user_last_timestamp.get(user_id)
        if previous is None:
            return None
        return max(0, timestamp - previous)


def run_sanity_baselines(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    split_name: str = "temporal_leave_one_out",
    eval_split: str = "test",
    ks: tuple[int, ...] = DEFAULT_KS,
    item_knn_neighbors: int = 50,
    item_knn_max_history_items: int = 100,
    cooccurrence_history_window: int = 200,
    use_validation_history_for_test: bool = True,
    exclude_seen: bool = True,
    seed: int = 2026,
    command: str = "tglrec evaluate sanity-baselines",
) -> SanityBaselineResult:
    """Evaluate popularity and item-kNN baselines on a processed dataset."""

    dataset_root = Path(dataset_dir)
    interactions_path = dataset_root / "interactions.csv"
    items_path = dataset_root / "items.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing processed interactions: {interactions_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing processed items: {items_path}")

    interactions = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path)
    split_col = _split_column(split_name)
    _validate_interactions(interactions, split_col)
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    if not ks:
        raise ValueError("ks must contain at least one cutoff")

    item_universe = {int(item_id) for item_id in items[schema.ITEM_ID].tolist()}
    train_events = _events_from_frame(interactions.loc[interactions[split_col] == "train"])
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
    rankings: dict[str, dict[int, list[int]]] = {"popularity": {}, "item_knn": {}}
    positives: dict[int, int] = {}
    segment_rows: list[dict[str, str]] = []

    stats = IncrementalTrainingStats()
    train_index = 0
    history_only_index = 0
    for case in sorted(eval_cases, key=lambda row: (row.timestamp, row.user_id, row.event_id)):
        while train_index < len(train_events) and train_events[train_index].timestamp < case.timestamp:
            train_case = train_events[train_index]
            stats.add_event(
                train_case.user_id,
                train_case.item_id,
                train_case.timestamp,
                cooccurrence_history_window=cooccurrence_history_window,
            )
            train_index += 1
        while history_only_index < len(history_only_events) and _event_precedes(
            history_only_events[history_only_index], case
        ):
            history_case = history_only_events[history_only_index]
            stats.add_user_history_event(
                history_case.user_id,
                history_case.item_id,
                history_case.timestamp,
            )
            history_only_index += 1

        candidates = set(item_universe)
        if exclude_seen:
            candidates.difference_update(stats.user_items.get(case.user_id, set()))
        candidates.add(case.item_id)

        case_key = case.event_id
        positives[case_key] = case.item_id
        rankings["popularity"][case_key] = [
            int(item_id) for item_id in rank_by_score(stats.popularity_scores(candidates))
        ][:max_k]
        rankings["item_knn"][case_key] = [
            int(item_id)
            for item_id in rank_by_score(
                stats.item_knn_scores(
                    case.user_id,
                    candidates,
                    neighbors=item_knn_neighbors,
                    max_history_items=item_knn_max_history_items,
                )
            )
        ][:max_k]
        segment_rows.append(_segment_row(case, stats))

    metrics = {
        baseline: evaluate_rankings(baseline_rankings, positives, ks=ks)
        for baseline, baseline_rankings in rankings.items()
    }
    run_root = Path(output_dir) if output_dir is not None else _default_run_dir()
    _write_run_outputs(
        run_root,
        command=command,
        config={
            "baseline_names": ["popularity", "item_knn"],
            "candidate_mode": "full_ranking",
            "dataset_dir": str(dataset_root),
            "eval_split": eval_split,
            "exclude_seen": exclude_seen,
            "cooccurrence_history_window": cooccurrence_history_window,
            "history_splits": _history_splits(
                eval_split,
                use_validation_history_for_test=use_validation_history_for_test,
            ),
            "item_knn_max_history_items": item_knn_max_history_items,
            "item_knn_neighbors": item_knn_neighbors,
            "ks": list(ks),
            "seed": seed,
            "split_name": split_name,
        },
        metrics=metrics,
        segment_rows=segment_rows,
        rankings=rankings,
        positives=positives,
        num_items=len(item_universe),
        num_eval_users=len({case.user_id for case in eval_cases}),
    )
    return SanityBaselineResult(output_dir=run_root, metrics=metrics, num_cases=len(eval_cases))


def _split_column(split_name: str) -> str:
    if split_name == "temporal_leave_one_out":
        return schema.SPLIT_LOO
    if split_name == "global_time":
        return schema.SPLIT_GLOBAL
    raise ValueError("split_name must be 'temporal_leave_one_out' or 'global_time'")


def _validate_interactions(interactions: pd.DataFrame, split_col: str) -> None:
    required = {
        schema.EVENT_ID,
        schema.USER_ID,
        schema.ITEM_ID,
        schema.TIMESTAMP,
        split_col,
    }
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"Missing required interaction columns: {missing}")


def _events_from_frame(frame: pd.DataFrame) -> list[EvaluationCase]:
    return _cases_from_frame(frame)


def _history_only_events(
    interactions: pd.DataFrame,
    *,
    split_col: str,
    eval_split: str,
    use_validation_history_for_test: bool,
) -> list[EvaluationCase]:
    if eval_split != "test" or not use_validation_history_for_test:
        return []
    return _cases_from_frame(interactions.loc[interactions[split_col] == "val"])


def _history_splits(eval_split: str, *, use_validation_history_for_test: bool) -> list[str]:
    if eval_split == "test" and use_validation_history_for_test:
        return ["train", "val_user_history_only"]
    return ["train"]


def _event_precedes(history_event: EvaluationCase, prediction_case: EvaluationCase) -> bool:
    if history_event.timestamp < prediction_case.timestamp:
        return True
    return (
        history_event.user_id == prediction_case.user_id
        and history_event.timestamp == prediction_case.timestamp
        and history_event.event_id < prediction_case.event_id
    )


def _cases_from_frame(frame: pd.DataFrame) -> list[EvaluationCase]:
    values = frame[
        [schema.USER_ID, schema.ITEM_ID, schema.TIMESTAMP, schema.EVENT_ID]
    ].to_numpy()
    cases = [
        EvaluationCase(
            user_id=int(user_id),
            item_id=int(item_id),
            timestamp=int(timestamp),
            event_id=int(event_id),
        )
        for user_id, item_id, timestamp, event_id in values
    ]
    return sorted(cases, key=lambda row: (row.timestamp, row.user_id, row.event_id))


def _segment_row(case: EvaluationCase, stats: IncrementalTrainingStats) -> dict[str, str]:
    target_popularity = stats.item_popularity.get(case.item_id, 0)
    last_gap = stats.last_gap_seconds(case.user_id, case.timestamp)
    return {
        "case_id": str(case.event_id),
        "user_id": str(case.user_id),
        "positive_item_id": str(case.item_id),
        "history_length_bucket": _history_bucket(stats.history_length(case.user_id)),
        "last_interaction_gap_bucket": _gap_bucket(last_gap),
        "target_popularity_bucket": _popularity_bucket(target_popularity),
        "target_cold_warm": "cold" if target_popularity == 0 else "warm",
        "semantic_vs_transition_case_type": "not_computed",
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
    return Path("runs") / f"{stamp}-sanity-baselines"


def _write_run_outputs(
    output_dir: Path,
    *,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    segment_rows: list[dict[str, str]],
    rankings: dict[str, dict[int, list[int]]],
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
            "num_eval_cases": len(positives),
            "num_eval_users": num_eval_users,
            "num_items": num_items,
        },
        root / "metrics.json",
    )
    _write_metrics_by_segment(root / "metrics_by_segment.csv", metrics, segment_rows, rankings, positives)
    _write_environment(root / "environment.json")
    (root / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    (root / "git_commit.txt").write_text(
        current_git_commit(".") + "\n", encoding="utf-8", newline="\n"
    )
    (root / "stdout.log").write_text(
        json.dumps({"output_dir": str(root), "metrics": metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "stderr.log").write_text("", encoding="utf-8", newline="\n")


def _write_metrics_by_segment(
    path: Path,
    metrics: dict[str, dict[str, float]],
    segment_rows: list[dict[str, str]],
    rankings: dict[str, dict[int, list[int]]],
    positives: dict[int, int],
) -> None:
    segment_names = [
        "history_length_bucket",
        "last_interaction_gap_bucket",
        "target_popularity_bucket",
        "target_cold_warm",
        "semantic_vs_transition_case_type",
    ]
    metric_names = sorted(next(iter(metrics.values())).keys())
    rows: list[dict[str, Any]] = []
    by_case = {int(row["case_id"]): row for row in segment_rows}
    for baseline, baseline_rankings in rankings.items():
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
                subset_rankings = {case_id: baseline_rankings[case_id] for case_id in case_ids}
                subset_positives = {case_id: positives[case_id] for case_id in case_ids}
                subset_metrics = evaluate_rankings(
                    subset_rankings,
                    subset_positives,
                    ks=tuple(int(name.split("@")[1]) for name in metric_names if name.startswith("HR@")),
                )
                output_row: dict[str, Any] = {
                    "baseline": baseline,
                    "segment_name": segment_name,
                    "segment_value": segment_value,
                    "num_cases": len(case_ids),
                }
                output_row.update(subset_metrics)
                rows.append(output_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline", "segment_name", "segment_value", "num_cases", *metric_names],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_environment(path: Path) -> None:
    write_json(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        path,
    )
