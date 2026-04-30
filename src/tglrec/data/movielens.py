"""MovieLens-1M preprocessing.

Official source checked 2026-04-29:
https://grouplens.org/datasets/movielens/1m/
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import (
    CHECKSUM_MANIFEST_NAME,
    COMMON_ARTIFACT_MANIFEST_FILES,
    PROCESSED_DATASET_FILES,
    build_checksum_manifest,
    same_user_timestamp_tie_stats,
    write_checksum_manifest,
)
from tglrec.data.splits import (
    apply_stable_ids,
    assert_no_future_leakage,
    assign_event_ids,
    global_time_split,
    iterative_min_filter,
    temporal_leave_one_out_split,
)
from tglrec.utils.io import ensure_dir
from tglrec.utils.logging import write_artifact_manifest

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass(frozen=True)
class PreprocessResult:
    """Summary of a completed preprocessing run."""

    output_dir: Path
    num_interactions: int
    num_users: int
    num_items: int
    metadata: dict[str, Any]


def preprocess_movielens_1m(
    *,
    output_dir: str | Path,
    raw_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    download: bool = False,
    download_dir: str | Path = "data/raw/movielens_1m",
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    global_train_ratio: float = 0.8,
    global_val_ratio: float = 0.1,
    seed: int = 2026,
) -> PreprocessResult:
    """Preprocess MovieLens-1M into normalized files and temporal splits."""

    raw_root = _prepare_raw_data(raw_dir=raw_dir, zip_path=zip_path, download=download, download_dir=download_dir)
    ratings = _read_ratings(raw_root / "ratings.dat")
    movies = _read_movies(raw_root / "movies.dat")

    filtered = iterative_min_filter(
        ratings,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    mapped, users, item_ids = apply_stable_ids(filtered)
    items = item_ids.merge(movies, on=schema.RAW_ITEM_ID, how="left", validate="one_to_one")
    items["title"] = items["title"].fillna("")
    items["genres"] = items["genres"].fillna("")

    interactions = assign_event_ids(mapped)
    interactions[schema.SPLIT_LOO] = temporal_leave_one_out_split(interactions)
    interactions[schema.SPLIT_GLOBAL], cutoffs = global_time_split(
        interactions, train_ratio=global_train_ratio, val_ratio=global_val_ratio
    )
    interactions = interactions[schema.INTERACTION_COLUMNS].copy()
    assert_no_future_leakage(interactions, schema.SPLIT_LOO)
    assert_no_future_leakage(interactions, schema.SPLIT_GLOBAL)

    root = ensure_dir(output_dir)
    interactions.to_csv(root / "interactions.csv", index=False)
    users.to_csv(root / "users.csv", index=False)
    items[[schema.ITEM_ID, schema.RAW_ITEM_ID, "title", "genres"]].to_csv(
        root / "items.csv", index=False
    )
    _write_split_files(interactions, root, schema.SPLIT_LOO, "temporal_leave_one_out")
    _write_split_files(interactions, root, schema.SPLIT_GLOBAL, "global_time")
    processed_files = PROCESSED_DATASET_FILES
    processed_file_checksums = build_checksum_manifest(root, processed_files)["files"]

    config = {
        "dataset": "movielens_1m",
        "source_url": MOVIELENS_1M_URL,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "global_train_ratio": global_train_ratio,
        "global_val_ratio": global_val_ratio,
        "seed": seed,
    }
    metadata: dict[str, Any] = {
        "dataset": "movielens_1m",
        "source_url": MOVIELENS_1M_URL,
        "source_page": "https://grouplens.org/datasets/movielens/1m/",
        "date_checked": "2026-04-29",
        "raw_root": str(raw_root),
        "num_interactions": int(len(interactions)),
        "num_users": int(users[schema.USER_ID].nunique()),
        "num_items": int(items[schema.ITEM_ID].nunique()),
        "min_timestamp": int(interactions[schema.TIMESTAMP].min()),
        "max_timestamp": int(interactions[schema.TIMESTAMP].max()),
        "global_time_cutoffs": {
            "train_end_exclusive": cutoffs.train_end,
            "val_end_exclusive": cutoffs.val_end,
        },
        "same_user_timestamp_tie_stats": same_user_timestamp_tie_stats(interactions),
        "temporal_loo_tie_policy": (
            "same-user identical timestamps are allowed for MovieLens-1M and broken "
            "deterministically by item/raw ids; tie counts are recorded for audit."
        ),
        "processed_file_checksums": processed_file_checksums,
        "files": {
            "interactions": "interactions.csv",
            "users": "users.csv",
            "items": "items.csv",
            "temporal_leave_one_out": "temporal_leave_one_out/{train,val,test}.csv",
            "global_time": "global_time/{train,val,test}.csv",
            "checksums": CHECKSUM_MANIFEST_NAME,
        },
    }
    command = " ".join(sys.argv) if sys.argv else "tglrec preprocess movielens-1m"
    write_artifact_manifest(root, command=command, config=config, metadata=metadata)
    write_checksum_manifest(root, [*processed_files, *COMMON_ARTIFACT_MANIFEST_FILES])
    return PreprocessResult(
        output_dir=root,
        num_interactions=len(interactions),
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        metadata=metadata,
    )


def _prepare_raw_data(
    *,
    raw_dir: str | Path | None,
    zip_path: str | Path | None,
    download: bool,
    download_dir: str | Path,
) -> Path:
    if raw_dir is not None:
        resolved = _resolve_raw_dir(Path(raw_dir))
        if resolved is not None:
            return resolved
    if zip_path is not None:
        return _extract_zip(Path(zip_path), Path(download_dir))
    if download:
        target_dir = ensure_dir(download_dir)
        archive = target_dir / "ml-1m.zip"
        if not archive.exists():
            try:
                urllib.request.urlretrieve(MOVIELENS_1M_URL, archive)
            except Exception as exc:  # pragma: no cover - network dependent
                raise RuntimeError(
                    "Could not download MovieLens-1M from the official GroupLens URL. "
                    "Use --zip-path or --raw-dir after following DATA_MANUAL_STEPS.md."
                ) from exc
        return _extract_zip(archive, target_dir)
    raise FileNotFoundError(
        "MovieLens-1M raw data not found. Provide --raw-dir, --zip-path, or use --download."
    )


def _resolve_raw_dir(path: Path) -> Path | None:
    candidates = [path, path / "ml-1m"]
    for candidate in candidates:
        if (candidate / "ratings.dat").exists() and (candidate / "movies.dat").exists():
            return candidate
    return None


def _extract_zip(zip_path: Path, output_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"MovieLens zip file does not exist: {zip_path}")
    output = ensure_dir(output_dir)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if not any(member.endswith("ratings.dat") for member in members):
            raise ValueError(f"Zip does not look like MovieLens-1M: {zip_path}")
        archive.extractall(output)
    resolved = _resolve_raw_dir(output)
    if resolved is None:
        raise FileNotFoundError(f"Could not find ratings.dat and movies.dat after extracting {zip_path}")
    return resolved


def _read_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing MovieLens ratings file: {path}")
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=[schema.RAW_USER_ID, schema.RAW_ITEM_ID, schema.RATING, schema.TIMESTAMP],
        encoding="latin-1",
    )
    ratings[schema.RAW_USER_ID] = ratings[schema.RAW_USER_ID].astype(str)
    ratings[schema.RAW_ITEM_ID] = ratings[schema.RAW_ITEM_ID].astype(str)
    ratings[schema.RATING] = ratings[schema.RATING].astype(float)
    ratings[schema.TIMESTAMP] = ratings[schema.TIMESTAMP].astype("int64")
    return ratings


def _read_movies(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing MovieLens movies file: {path}")
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=[schema.RAW_ITEM_ID, "title", "genres"],
        encoding="latin-1",
    )
    movies[schema.RAW_ITEM_ID] = movies[schema.RAW_ITEM_ID].astype(str)
    return movies


def _write_split_files(interactions: pd.DataFrame, root: Path, split_col: str, split_name: str) -> None:
    split_dir = ensure_dir(root / split_name)
    for split in ("train", "val", "test"):
        frame = interactions.loc[interactions[split_col] == split].drop(columns=[schema.SPLIT_LOO, schema.SPLIT_GLOBAL])
        frame.to_csv(split_dir / f"{split}.csv", index=False)


def copy_manual_steps(path: str | Path) -> None:
    """Copy the repository manual data steps into an artifact directory when useful."""

    source = Path("DATA_MANUAL_STEPS.md")
    if source.exists():
        shutil.copyfile(source, Path(path) / "DATA_MANUAL_STEPS.md")
