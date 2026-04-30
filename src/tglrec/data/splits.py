"""Deterministic preprocessing and temporal split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tglrec.data import schema


@dataclass(frozen=True)
class GlobalTimeCutoffs:
    """Timestamp boundaries for global-time split."""

    train_end: int
    val_end: int


def iterative_min_filter(
    interactions: pd.DataFrame,
    *,
    min_user_interactions: int,
    min_item_interactions: int,
    user_col: str = schema.RAW_USER_ID,
    item_col: str = schema.RAW_ITEM_ID,
) -> pd.DataFrame:
    """Filter interactions until every remaining user and item meets thresholds."""

    if min_user_interactions <= 0 or min_item_interactions <= 0:
        raise ValueError("Minimum interaction thresholds must be positive.")

    filtered = interactions.copy()
    while True:
        before = len(filtered)
        user_counts = filtered[user_col].value_counts(sort=False)
        keep_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered[user_col].isin(keep_users)]
        item_counts = filtered[item_col].value_counts(sort=False)
        keep_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered[item_col].isin(keep_items)]
        if len(filtered) == before:
            break
        if filtered.empty:
            raise ValueError(
                "Filtering removed all interactions; lower min_user_interactions or "
                "min_item_interactions."
            )
    return filtered.copy()


def stable_id_map(values: pd.Series) -> dict[str, int]:
    """Map raw identifiers to contiguous integer ids independent of row order."""

    unique = sorted({str(value) for value in values})
    return {raw_id: idx for idx, raw_id in enumerate(unique)}


def apply_stable_ids(
    interactions: pd.DataFrame,
    *,
    user_col: str = schema.RAW_USER_ID,
    item_col: str = schema.RAW_ITEM_ID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach deterministic user_id and item_id columns and return mapping tables."""

    user_map = stable_id_map(interactions[user_col])
    item_map = stable_id_map(interactions[item_col])
    mapped = interactions.copy()
    mapped[user_col] = mapped[user_col].astype(str)
    mapped[item_col] = mapped[item_col].astype(str)
    mapped[schema.USER_ID] = mapped[user_col].map(user_map).astype("int64")
    mapped[schema.ITEM_ID] = mapped[item_col].map(item_map).astype("int64")
    users = pd.DataFrame(
        [{schema.RAW_USER_ID: raw_id, schema.USER_ID: idx} for raw_id, idx in user_map.items()]
    )
    items = pd.DataFrame(
        [{schema.RAW_ITEM_ID: raw_id, schema.ITEM_ID: idx} for raw_id, idx in item_map.items()]
    )
    return mapped, users, items


def sort_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Sort interactions by user, time, item, and raw ids to break timestamp ties."""

    return interactions.sort_values(
        [
            schema.USER_ID,
            schema.TIMESTAMP,
            schema.ITEM_ID,
            schema.RAW_ITEM_ID,
            schema.RAW_USER_ID,
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def assign_event_ids(interactions: pd.DataFrame) -> pd.DataFrame:
    """Assign stable event ids after deterministic sorting."""

    sorted_df = sort_interactions(interactions).copy()
    sorted_df[schema.EVENT_ID] = range(len(sorted_df))
    return sorted_df


def temporal_leave_one_out_split(interactions: pd.DataFrame) -> pd.Series:
    """Assign train/val/test labels using the last two events per user."""

    labels = pd.Series("train", index=interactions.index, dtype="object")
    for _, group in interactions.groupby(schema.USER_ID, sort=True):
        ordered_index = group.sort_values(
            [schema.TIMESTAMP, schema.ITEM_ID, schema.EVENT_ID], kind="mergesort"
        ).index
        if len(ordered_index) < 3:
            raise ValueError("Temporal leave-one-out requires at least 3 interactions per user.")
        labels.loc[ordered_index[-2]] = "val"
        labels.loc[ordered_index[-1]] = "test"
    return labels


def global_time_split(
    interactions: pd.DataFrame,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.Series, GlobalTimeCutoffs]:
    """Assign train/val/test labels by global event time.

    The cutoff positions are event-count based after a stable timestamp sort. Labels are then
    applied by timestamp boundaries, so all train events occur before validation events and all
    validation events occur before test events. Ties at the cutoff are kept on the later side to
    avoid mixing equal timestamps across train and validation/test.
    """

    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    ordered = interactions.sort_values(
        [schema.TIMESTAMP, schema.USER_ID, schema.ITEM_ID, schema.EVENT_ID], kind="mergesort"
    )
    n_events = len(ordered)
    if n_events < 3:
        raise ValueError("Global-time split requires at least 3 interactions.")
    train_pos = max(1, min(n_events - 2, int(n_events * train_ratio)))
    val_pos = max(train_pos + 1, min(n_events - 1, int(n_events * (train_ratio + val_ratio))))
    train_end = int(ordered.iloc[train_pos][schema.TIMESTAMP])
    val_end = int(ordered.iloc[val_pos][schema.TIMESTAMP])

    labels = pd.Series("test", index=interactions.index, dtype="object")
    labels.loc[interactions[schema.TIMESTAMP] < train_end] = "train"
    labels.loc[
        (interactions[schema.TIMESTAMP] >= train_end) & (interactions[schema.TIMESTAMP] < val_end)
    ] = "val"
    cutoffs = GlobalTimeCutoffs(train_end=train_end, val_end=val_end)
    return labels, cutoffs


def assert_no_future_leakage(interactions: pd.DataFrame, split_col: str) -> None:
    """Validate temporal ordering constraints for a processed split column."""

    if split_col == schema.SPLIT_LOO:
        for user_id, group in interactions.groupby(schema.USER_ID, sort=True):
            train_max = group.loc[group[split_col] == "train", schema.TIMESTAMP].max()
            val_times = group.loc[group[split_col] == "val", schema.TIMESTAMP]
            test_times = group.loc[group[split_col] == "test", schema.TIMESTAMP]
            if len(val_times) != 1 or len(test_times) != 1:
                raise ValueError(f"User {user_id} must have exactly one val and one test event.")
            if pd.notna(train_max) and train_max > val_times.iloc[0]:
                raise ValueError(f"Train event after validation for user {user_id}.")
            if val_times.iloc[0] > test_times.iloc[0]:
                raise ValueError(f"Validation event after test for user {user_id}.")
        return

    if split_col == schema.SPLIT_GLOBAL:
        train = interactions.loc[interactions[split_col] == "train", schema.TIMESTAMP]
        val = interactions.loc[interactions[split_col] == "val", schema.TIMESTAMP]
        test = interactions.loc[interactions[split_col] == "test", schema.TIMESTAMP]
        if train.empty or val.empty or test.empty:
            raise ValueError("Global-time split must produce non-empty train, val, and test sets.")
        if train.max() >= val.min():
            raise ValueError("Global-time train overlaps validation or later timestamps.")
        if val.max() >= test.min():
            raise ValueError("Global-time validation overlaps test or later timestamps.")
        return

    raise ValueError(f"Unknown split column: {split_col}")


def training_events_as_of(
    interactions: pd.DataFrame,
    *,
    split_col: str,
    prediction_timestamp: int,
    strict: bool = True,
) -> pd.DataFrame:
    """Return training events available at a prediction timestamp.

    Per-user leave-one-out is useful for controlled sequential tests, but a global
    training table can still contain another user's later event. Temporal graph,
    popularity, co-occurrence, and diagnostic candidate builders should call this
    helper when their evidence is meant to be available as of a target event.
    """

    if split_col not in interactions.columns:
        raise ValueError(f"Missing split column: {split_col}")
    train = interactions[interactions[split_col] == "train"]
    if strict:
        return train[train[schema.TIMESTAMP] < prediction_timestamp].copy()
    return train[train[schema.TIMESTAMP] <= prediction_timestamp].copy()
