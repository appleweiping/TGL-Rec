"""Deterministic NumPy BPR-MF baseline for implicit recommendation."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import write_checksum_manifest
from tglrec.eval.metrics import evaluate_rankings, rank_by_score
from tglrec.models.sanity_baselines import (
    DEFAULT_KS,
    EvaluationCase,
    _cases_from_frame,
    _event_precedes,
    _gap_bucket,
    _history_bucket,
    _history_only_events,
    _history_splits,
    _popularity_bucket,
    _split_column,
    _validate_interactions,
)
from tglrec.utils.config import write_config
from tglrec.utils.io import ensure_dir, write_json
from tglrec.utils.logging import current_git_commit


@dataclass(frozen=True)
class BPRMFResult:
    """Summary of a completed BPR-MF run."""

    output_dir: Path
    metrics: dict[str, float]
    num_cases: int


@dataclass(frozen=True)
class _PositivePair:
    user_id: int
    item_id: int


@dataclass
class _BPRModel:
    user_factors: np.ndarray
    item_factors: np.ndarray
    item_bias: np.ndarray
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]
    index_to_item: list[int]


def run_bpr_mf(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    split_name: str = "temporal_leave_one_out",
    eval_split: str = "test",
    ks: tuple[int, ...] = DEFAULT_KS,
    factors: int = 64,
    epochs: int = 20,
    learning_rate: float = 0.05,
    regularization: float = 0.0025,
    max_train_pairs: int | None = None,
    max_eval_cases: int | None = None,
    use_validation_history_for_test: bool = True,
    exclude_seen: bool = True,
    seed: int = 2026,
    command: str = "tglrec train bpr-mf",
) -> BPRMFResult:
    """Train and evaluate a local BPR matrix-factorization baseline."""

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
    if factors <= 0:
        raise ValueError("factors must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if max_train_pairs is not None and max_train_pairs <= 0:
        raise ValueError("max_train_pairs must be positive when provided")
    if max_eval_cases is not None and max_eval_cases <= 0:
        raise ValueError("max_eval_cases must be positive when provided")

    interactions = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path)
    split_col = _split_column(split_name)
    _validate_interactions(interactions, split_col)

    item_universe = sorted({int(item_id) for item_id in items[schema.ITEM_ID].tolist()})
    interaction_items = {int(item_id) for item_id in interactions[schema.ITEM_ID].tolist()}
    missing_items = sorted(interaction_items - set(item_universe))
    if missing_items:
        preview = ", ".join(str(item_id) for item_id in missing_items[:5])
        raise ValueError(
            f"items.csv is missing {len(missing_items)} interaction item ids, e.g. {preview}"
        )

    train_frame = interactions.loc[interactions[split_col] == "train"].copy()
    history_only_events = _history_only_events(
        interactions,
        split_col=split_col,
        eval_split=eval_split,
        use_validation_history_for_test=use_validation_history_for_test,
    )
    eval_cases = _cases_from_frame(interactions.loc[interactions[split_col] == eval_split])
    if max_eval_cases is not None:
        eval_cases = eval_cases[:max_eval_cases]
    if not eval_cases:
        raise ValueError(f"No {eval_split} cases found in {interactions_path}")

    positives = _positive_pairs_from_frame(train_frame)
    if max_train_pairs is not None:
        positives = positives[:max_train_pairs]
    if not positives:
        raise ValueError("No train user-item pairs are available for BPR-MF.")

    train_seen_by_user = _seen_items_by_user_from_frame(train_frame)
    train_last_timestamp_by_user = _last_timestamp_by_user_from_frame(train_frame)
    item_popularity = _item_popularity_from_frame(train_frame)
    model, epoch_rows = _train_bpr(
        positives,
        item_universe=item_universe,
        train_seen_by_user=train_seen_by_user,
        factors=factors,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=seed,
    )
    rankings, positive_by_case, case_rows, segment_rows = _evaluate_bpr(
        model,
        eval_cases,
        history_only_events=history_only_events,
        train_seen_by_user=train_seen_by_user,
        train_last_timestamp_by_user=train_last_timestamp_by_user,
        item_universe=item_universe,
        item_popularity=item_popularity,
        exclude_seen=exclude_seen,
        ks=ks,
    )
    metrics = evaluate_rankings(rankings, positive_by_case, ks=ks)
    run_root = Path(output_dir) if output_dir is not None else _default_run_dir()
    _write_run_outputs(
        run_root,
        command=command,
        config={
            "baseline_name": "bpr_mf",
            "candidate_mode": "full_ranking",
            "dataset_dir": str(dataset_root),
            "dataset_provenance": _dataset_provenance(dataset_root),
            "epochs": epochs,
            "eval_split": eval_split,
            "exclude_seen": exclude_seen,
            "factors": factors,
            "history_splits": _history_splits(
                eval_split,
                use_validation_history_for_test=use_validation_history_for_test,
            ),
            "ks": list(ks),
            "learning_rate": learning_rate,
            "max_eval_cases": max_eval_cases,
            "max_train_pairs": max_train_pairs,
            "num_train_pairs": len(positives),
            "regularization": regularization,
            "seed": seed,
            "split_name": split_name,
            "training_policy": (
                "BPR-MF is trained only on split=train user-item pairs. Validation events are "
                "never used as optimization positives; for test evaluation they can only be used "
                "as prior seen history for candidate filtering when enabled."
            ),
        },
        metrics=metrics,
        epoch_rows=epoch_rows,
        case_rows=case_rows,
        segment_rows=segment_rows,
        rankings=rankings,
        positives=positive_by_case,
        num_eval_users=len({case.user_id for case in eval_cases}),
        num_items=len(item_universe),
    )
    return BPRMFResult(output_dir=run_root, metrics=metrics, num_cases=len(eval_cases))


def _positive_pairs_from_frame(train_frame: pd.DataFrame) -> list[_PositivePair]:
    ordered = train_frame.sort_values(
        [schema.TIMESTAMP, schema.USER_ID, schema.EVENT_ID],
        kind="mergesort",
    )
    deduplicated = ordered.drop_duplicates(
        subset=[schema.USER_ID, schema.ITEM_ID],
        keep="first",
    )
    values = deduplicated[[schema.USER_ID, schema.ITEM_ID]].to_numpy()
    return [
        _PositivePair(user_id=int(user_id), item_id=int(item_id))
        for user_id, item_id in values
    ]


def _seen_items_by_user_from_frame(train_frame: pd.DataFrame) -> dict[int, set[int]]:
    grouped = train_frame.groupby(schema.USER_ID, sort=False)[schema.ITEM_ID].unique()
    return {
        int(user_id): {int(item_id) for item_id in item_ids}
        for user_id, item_ids in grouped.items()
    }


def _last_timestamp_by_user_from_frame(train_frame: pd.DataFrame) -> dict[int, int]:
    grouped = train_frame.groupby(schema.USER_ID, sort=False)[schema.TIMESTAMP].max()
    return {int(user_id): int(timestamp) for user_id, timestamp in grouped.items()}


def _item_popularity_from_frame(train_frame: pd.DataFrame) -> Counter[int]:
    counts = train_frame[schema.ITEM_ID].value_counts(sort=False)
    return Counter({int(item_id): int(count) for item_id, count in counts.items()})


def _train_bpr(
    positives: list[_PositivePair],
    *,
    item_universe: list[int],
    train_seen_by_user: dict[int, set[int]],
    factors: int,
    epochs: int,
    learning_rate: float,
    regularization: float,
    seed: int,
) -> tuple[_BPRModel, list[dict[str, Any]]]:
    users = sorted({pair.user_id for pair in positives})
    user_to_index = {user_id: index for index, user_id in enumerate(users)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_universe)}
    rng = np.random.default_rng(seed)
    scale = 0.1 / float(factors) ** 0.5
    user_factors = rng.normal(0.0, scale, size=(len(users), factors)).astype(np.float64)
    item_factors = rng.normal(0.0, scale, size=(len(item_universe), factors)).astype(np.float64)
    item_bias = np.zeros(len(item_universe), dtype=np.float64)
    positive_indices = np.array(
        [(user_to_index[pair.user_id], item_to_index[pair.item_id]) for pair in positives],
        dtype=np.int64,
    )
    user_seen_indices = {
        user_to_index[user_id]: frozenset(item_to_index[item_id] for item_id in item_ids)
        for user_id, item_ids in train_seen_by_user.items()
        if user_id in user_to_index
    }
    all_item_indices = np.arange(len(item_universe), dtype=np.int64)
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(positive_indices))
        total_loss = 0.0
        updates = 0
        for row_index in order:
            user_index, positive_item_index = positive_indices[row_index]
            negative_item_index = _sample_negative_item(
                rng,
                all_item_indices=all_item_indices,
                seen_items=user_seen_indices.get(int(user_index), frozenset()),
            )
            if negative_item_index is None:
                continue
            user_vector = user_factors[user_index].copy()
            positive_vector = item_factors[positive_item_index].copy()
            negative_vector = item_factors[negative_item_index].copy()
            margin = (
                float(user_vector @ (positive_vector - negative_vector))
                + float(item_bias[positive_item_index] - item_bias[negative_item_index])
            )
            coefficient = _sigmoid_negative(margin)
            user_factors[user_index] += learning_rate * (
                coefficient * (positive_vector - negative_vector)
                - regularization * user_vector
            )
            item_factors[positive_item_index] += learning_rate * (
                coefficient * user_vector - regularization * positive_vector
            )
            item_factors[negative_item_index] += learning_rate * (
                -coefficient * user_vector - regularization * negative_vector
            )
            item_bias[positive_item_index] += learning_rate * (
                coefficient - regularization * item_bias[positive_item_index]
            )
            item_bias[negative_item_index] += learning_rate * (
                -coefficient - regularization * item_bias[negative_item_index]
            )
            total_loss += _softplus_negative_margin(margin)
            updates += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "mean_bpr_loss": 0.0 if updates == 0 else total_loss / updates,
                "updates": updates,
            }
        )
    return (
        _BPRModel(
            user_factors=user_factors,
            item_factors=item_factors,
            item_bias=item_bias,
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            index_to_item=list(item_universe),
        ),
        epoch_rows,
    )


def _sample_negative_item(
    rng: np.random.Generator,
    *,
    all_item_indices: np.ndarray,
    seen_items: frozenset[int],
) -> int | None:
    if len(seen_items) >= len(all_item_indices):
        return None
    while True:
        item_index = int(rng.integers(0, len(all_item_indices)))
        if item_index not in seen_items:
            return item_index


def _sigmoid_negative(value: float) -> float:
    if value >= 0:
        exp_negative = np.exp(-value)
        return float(exp_negative / (1.0 + exp_negative))
    exp_positive = np.exp(value)
    return float(1.0 / (1.0 + exp_positive))


def _softplus_negative_margin(value: float) -> float:
    return float(np.logaddexp(0.0, -value))


def _evaluate_bpr(
    model: _BPRModel,
    eval_cases: list[EvaluationCase],
    *,
    history_only_events: list[EvaluationCase],
    train_seen_by_user: dict[int, set[int]],
    train_last_timestamp_by_user: dict[int, int],
    item_universe: list[int],
    item_popularity: Counter[int],
    exclude_seen: bool,
    ks: tuple[int, ...],
) -> tuple[
    dict[int, list[int]],
    dict[int, int],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    rankings: dict[int, list[int]] = {}
    positives: dict[int, int] = {}
    case_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, str]] = []
    seen_by_user = {user_id: set(items) for user_id, items in train_seen_by_user.items()}
    last_timestamp_by_user = dict(train_last_timestamp_by_user)
    history_index = 0
    sorted_cases = sorted(eval_cases, key=lambda row: (row.timestamp, row.user_id, row.event_id))
    max_k = max(ks)
    for case in sorted_cases:
        while history_index < len(history_only_events) and _event_precedes(
            history_only_events[history_index], case
        ):
            history = history_only_events[history_index]
            seen_by_user.setdefault(history.user_id, set()).add(history.item_id)
            previous_timestamp = last_timestamp_by_user.get(history.user_id)
            if previous_timestamp is None or history.timestamp > previous_timestamp:
                last_timestamp_by_user[history.user_id] = history.timestamp
            history_index += 1
        candidates = set(item_universe)
        user_seen = seen_by_user.get(case.user_id, set())
        if exclude_seen:
            candidates.difference_update(user_seen)
        candidates.add(case.item_id)
        scores = _bpr_scores(model, case.user_id, candidates)
        ranked_items = [int(item_id) for item_id in rank_by_score(scores)]
        rankings[case.event_id] = ranked_items[:max_k]
        positives[case.event_id] = case.item_id
        target_rank = _positive_rank(ranked_items, case.item_id)
        case_rows.append(
            {
                "case_id": case.event_id,
                "user_id": case.user_id,
                "target_item_id": case.item_id,
                "target_timestamp": case.timestamp,
                "target_rank": "" if target_rank is None else target_rank,
                "target_score": scores.get(case.item_id, 0.0),
                "candidate_count": len(candidates),
                "top_item_ids_json": json.dumps(ranked_items[:max_k]),
            }
        )
        segment_rows.append(
            _segment_row(
                case,
                history_items=user_seen,
                last_history_timestamp=last_timestamp_by_user.get(case.user_id),
                target_popularity=item_popularity.get(case.item_id, 0),
            )
        )
    return rankings, positives, case_rows, segment_rows


def _bpr_scores(model: _BPRModel, user_id: int, candidates: set[int]) -> dict[int, float]:
    user_index = model.user_to_index.get(user_id)
    if user_index is None:
        user_vector = np.zeros(model.item_factors.shape[1], dtype=np.float64)
    else:
        user_vector = model.user_factors[user_index]
    scores: dict[int, float] = {}
    for item_id in candidates:
        item_index = model.item_to_index.get(item_id)
        if item_index is None:
            continue
        scores[item_id] = float(user_vector @ model.item_factors[item_index] + model.item_bias[item_index])
    return scores


def _positive_rank(ranked_items: list[int], positive_item: int) -> int | None:
    for rank, item_id in enumerate(ranked_items, start=1):
        if item_id == positive_item:
            return rank
    return None


def _segment_row(
    case: EvaluationCase,
    *,
    history_items: set[int],
    last_history_timestamp: int | None,
    target_popularity: int,
) -> dict[str, str]:
    last_gap = (
        None
        if last_history_timestamp is None
        else max(0, case.timestamp - last_history_timestamp)
    )
    return {
        "case_id": str(case.event_id),
        "user_id": str(case.user_id),
        "positive_item_id": str(case.item_id),
        "history_length_bucket": _history_bucket(len(history_items)),
        "last_interaction_gap_bucket": _gap_bucket(last_gap),
        "target_popularity_bucket": _popularity_bucket(target_popularity),
        "target_cold_warm": "cold" if target_popularity == 0 else "warm",
        "semantic_vs_transition_case_type": "not_computed",
    }


def _default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{stamp}-bpr-mf"


def _write_run_outputs(
    output_dir: Path,
    *,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    epoch_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, str]],
    rankings: dict[int, list[int]],
    positives: dict[int, int],
    num_eval_users: int,
    num_items: int,
) -> None:
    root = ensure_dir(output_dir)
    write_config(config, root / "config.yaml")
    write_json(
        {
            "baseline": config["baseline_name"],
            "candidate_mode": config["candidate_mode"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "num_eval_cases": len(positives),
            "num_eval_users": num_eval_users,
            "num_items": num_items,
        },
        root / "metrics.json",
    )
    _write_metrics_by_epoch(root / "metrics_by_epoch.csv", epoch_rows)
    _write_metrics_by_case(root / "metrics_by_case.csv", case_rows, ks=tuple(config["ks"]))
    _write_metrics_by_segment(root / "metrics_by_segment.csv", metrics, segment_rows, rankings, positives)
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
            "metrics_by_epoch.csv",
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


def _write_metrics_by_epoch(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mean_bpr_loss", "updates"])
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_by_case(path: Path, rows: list[dict[str, Any]], *, ks: tuple[int, ...]) -> None:
    fieldnames = [
        "case_id",
        "user_id",
        "target_item_id",
        "target_timestamp",
        "target_rank",
        "target_score",
        "candidate_count",
        "top_item_ids_json",
        *[f"HR@{k}" for k in sorted(set(ks))],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (int(value["target_timestamp"]), int(value["case_id"]))):
            target_rank = None if row["target_rank"] == "" else int(row["target_rank"])
            output = dict(row)
            for k in sorted(set(ks)):
                output[f"HR@{k}"] = int(target_rank is not None and target_rank <= k)
            writer.writerow(output)


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
    metric_names = sorted(metrics)
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
            subset_metrics = evaluate_rankings(
                subset_rankings,
                subset_positives,
                ks=tuple(int(name.split("@")[1]) for name in metric_names if name.startswith("HR@")),
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
    from tglrec.data.artifacts import CHECKSUM_MANIFEST_NAME, file_fingerprint

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
