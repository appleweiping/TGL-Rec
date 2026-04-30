"""Shared reproducibility helpers for processed dataset artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from tglrec.data import schema
from tglrec.utils.io import write_json

CHECKSUM_MANIFEST_NAME = "checksums.json"
STANDARD_PROCESSED_DATASET_FILES = [
    "interactions.csv",
    "users.csv",
    "items.csv",
    "temporal_leave_one_out/train.csv",
    "temporal_leave_one_out/val.csv",
    "temporal_leave_one_out/test.csv",
    "global_time/train.csv",
    "global_time/val.csv",
    "global_time/test.csv",
]
COMMON_MANIFEST_FILES = [
    "config.yaml",
    "metadata.json",
    "command.txt",
    "git_commit.txt",
    "created_at_utc.txt",
]
PROCESSED_DATASET_FILES = [
    "interactions.csv",
    "users.csv",
    "items.csv",
    "temporal_leave_one_out/train.csv",
    "temporal_leave_one_out/val.csv",
    "temporal_leave_one_out/test.csv",
    "global_time/train.csv",
    "global_time/val.csv",
    "global_time/test.csv",
]
COMMON_ARTIFACT_MANIFEST_FILES = [
    "config.yaml",
    "metadata.json",
    "command.txt",
    "git_commit.txt",
    "created_at_utc.txt",
]


def same_user_timestamp_tie_stats(
    interactions: pd.DataFrame,
    *,
    user_col: str = schema.RAW_USER_ID,
    timestamp_col: str = schema.TIMESTAMP,
) -> dict[str, int]:
    """Return aggregate counts for same-user events sharing an identical timestamp."""

    if user_col not in interactions.columns:
        raise ValueError(f"Missing user column for timestamp tie stats: {user_col}")
    if timestamp_col not in interactions.columns:
        raise ValueError(f"Missing timestamp column for timestamp tie stats: {timestamp_col}")
    if interactions.empty:
        return {
            "tied_groups": 0,
            "tied_rows": 0,
            "tied_extra_rows": 0,
            "affected_users": 0,
            "max_events_at_same_timestamp": 0,
        }

    counts = interactions.groupby([user_col, timestamp_col], sort=False).size()
    tied = counts[counts > 1]
    if tied.empty:
        return {
            "tied_groups": 0,
            "tied_rows": 0,
            "tied_extra_rows": 0,
            "affected_users": 0,
            "max_events_at_same_timestamp": 1,
        }
    affected_users = tied.index.get_level_values(0).nunique()
    return {
        "tied_groups": int(len(tied)),
        "tied_rows": int(tied.sum()),
        "tied_extra_rows": int((tied - 1).sum()),
        "affected_users": int(affected_users),
        "max_events_at_same_timestamp": int(tied.max()),
    }


def file_fingerprint(path: str | Path, *, include_path: bool = False) -> dict[str, Any]:
    """Return a SHA256 fingerprint for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    fingerprint: dict[str, Any] = {
        "bytes": int(file_path.stat().st_size),
        "sha256": digest.hexdigest(),
    }
    if include_path:
        fingerprint = {"path": str(file_path), **fingerprint}
    return fingerprint


def build_checksum_manifest(root: str | Path, relative_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Build a deterministic checksum manifest for files under an artifact root."""

    root_path = Path(root)
    files: dict[str, dict[str, Any]] = {}
    for relative in sorted({Path(path).as_posix() for path in relative_paths}):
        if relative == CHECKSUM_MANIFEST_NAME:
            raise ValueError(f"{CHECKSUM_MANIFEST_NAME} cannot include itself.")
        file_path = root_path / relative
        if not file_path.is_file():
            raise FileNotFoundError(f"Cannot checksum missing artifact file: {file_path}")
        files[relative] = file_fingerprint(file_path)
    return {
        "manifest_version": 1,
        "algorithm": "sha256",
        "files": files,
    }


def write_checksum_manifest(
    root: str | Path,
    relative_paths: Iterable[str | Path],
    *,
    manifest_name: str = CHECKSUM_MANIFEST_NAME,
) -> dict[str, Any]:
    """Write and return a checksum manifest for artifact files under ``root``."""

    manifest = build_checksum_manifest(root, relative_paths)
    write_json(manifest, Path(root) / manifest_name)
    return manifest
