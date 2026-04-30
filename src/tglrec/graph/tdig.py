"""Temporal directed item graph construction and direct retrieval.

The TDIG builder in this module uses only observed consecutive transitions from
training events. It is deliberately deterministic: event ordering, edge rows,
and retrieval ties all have explicit sort keys.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import CHECKSUM_MANIFEST_NAME, file_fingerprint, write_checksum_manifest
from tglrec.utils.io import ensure_dir, read_json, write_json
from tglrec.utils.logging import current_git_commit, write_artifact_manifest

SAME_SESSION_SECONDS = 30 * 60
DAY_SECONDS = 24 * 60 * 60
WEEK_SECONDS = 7 * DAY_SECONDS
MONTH_SECONDS = 30 * DAY_SECONDS

GAP_BUCKETS = ("same_session", "within_1d", "within_1w", "within_1m", "long_gap")
TDIG_ARTIFACT_FILES = [
    "edges.csv",
    "metadata.json",
    "config.yaml",
    "command.txt",
    "git_commit.txt",
    "created_at_utc.txt",
    "stdout.log",
    "stderr.log",
    "environment.json",
]


@dataclass(frozen=True)
class EdgeStats:
    """Statistics for one directed item-to-item transition edge."""

    source_item_id: int
    target_item_id: int
    support: int
    source_transition_count: int
    target_transition_count: int
    total_transition_count: int
    transition_probability: float
    lift: float
    pmi: float
    direction_asymmetry: float
    first_transition_timestamp: int
    last_transition_timestamp: int
    mean_transition_timestamp: float
    mean_gap_seconds: float
    gap_histogram: dict[str, int]

    def to_row(self) -> dict[str, Any]:
        """Return a stable CSV/JSON-friendly edge record."""

        row: dict[str, Any] = {
            "source_item_id": self.source_item_id,
            "target_item_id": self.target_item_id,
            "support": self.support,
            "source_transition_count": self.source_transition_count,
            "target_transition_count": self.target_transition_count,
            "total_transition_count": self.total_transition_count,
            "transition_probability": self.transition_probability,
            "lift": self.lift,
            "pmi": self.pmi,
            "direction_asymmetry": self.direction_asymmetry,
            "first_transition_timestamp": self.first_transition_timestamp,
            "last_transition_timestamp": self.last_transition_timestamp,
            "mean_transition_timestamp": self.mean_transition_timestamp,
            "mean_gap_seconds": self.mean_gap_seconds,
        }
        for bucket in GAP_BUCKETS:
            row[f"gap_{bucket}"] = self.gap_histogram.get(bucket, 0)
        row["gap_histogram_json"] = json.dumps(self.gap_histogram, sort_keys=True)
        return row


@dataclass(frozen=True)
class DirectTransitionCandidate:
    """One auditable direct retrieval result from the TDIG."""

    source_item_id: int
    target_item_id: int
    score: float
    edge: EdgeStats

    def to_dict(self) -> dict[str, Any]:
        row = self.edge.to_row()
        row["score"] = self.score
        return row


@dataclass(frozen=True)
class TDIGBuildResult:
    """Summary of a TDIG artifact build."""

    output_dir: Path
    num_edges: int
    num_transitions: int
    metadata: dict[str, Any]


class TemporalDirectedItemGraph:
    """In-memory TDIG with deterministic direct-transition retrieval."""

    def __init__(self, edges: Iterable[EdgeStats]) -> None:
        self.edges: dict[tuple[int, int], EdgeStats] = {
            (edge.source_item_id, edge.target_item_id): edge for edge in edges
        }
        self._outgoing: dict[int, list[EdgeStats]] = defaultdict(list)
        for edge in self.edges.values():
            self._outgoing[edge.source_item_id].append(edge)
        for source_item_id, source_edges in self._outgoing.items():
            self._outgoing[source_item_id] = sorted(source_edges, key=_edge_identity_key)

    def retrieve_direct(
        self,
        source_item_id: int,
        *,
        top_k: int = 50,
        score_field: str = "transition_probability",
        gap_bucket: str | None = None,
    ) -> list[DirectTransitionCandidate]:
        """Return direct next-item candidates from one source item.

        ``gap_bucket`` can be used to score by support within one time-gap bucket while
        still returning the full edge statistics for audit.
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if gap_bucket is not None and gap_bucket not in GAP_BUCKETS:
            raise ValueError(f"Unknown gap bucket: {gap_bucket}")
        source_edges = self._outgoing.get(int(source_item_id), [])
        candidates = [
            DirectTransitionCandidate(
                source_item_id=edge.source_item_id,
                target_item_id=edge.target_item_id,
                score=_edge_score(edge, score_field=score_field, gap_bucket=gap_bucket),
                edge=edge,
            )
            for edge in source_edges
        ]
        if gap_bucket is not None:
            candidates = [candidate for candidate in candidates if candidate.score > 0]
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                -candidate.edge.support,
                -candidate.edge.last_transition_timestamp,
                candidate.target_item_id,
            ),
        )[:top_k]

    def to_edge_rows(self) -> list[dict[str, Any]]:
        """Return all edge rows in stable source/target order."""

        return [self.edges[key].to_row() for key in sorted(self.edges)]


def gap_bucket(gap_seconds: int | float) -> str:
    """Map a non-negative transition gap in seconds to the TDIG bucket name."""

    if gap_seconds < 0:
        raise ValueError(f"Transition gap must be non-negative, got {gap_seconds}")
    if gap_seconds <= SAME_SESSION_SECONDS:
        return "same_session"
    if gap_seconds <= DAY_SECONDS:
        return "within_1d"
    if gap_seconds <= WEEK_SECONDS:
        return "within_1w"
    if gap_seconds <= MONTH_SECONDS:
        return "within_1m"
    return "long_gap"


def build_tdig_from_processed_split(
    *,
    dataset_dir: str | Path,
    split_name: str = "temporal_leave_one_out",
    train_split_label: str = "train",
    strict_before_timestamp: int | None = None,
    include_same_timestamp_transitions: bool = False,
) -> tuple[TemporalDirectedItemGraph, dict[str, Any]]:
    """Build a train-only TDIG from a processed dataset artifact."""

    dataset_root = Path(dataset_dir)
    interactions_path = dataset_root / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing processed interactions: {interactions_path}")
    interactions = pd.read_csv(interactions_path)
    split_col = _split_column(split_name)
    graph, metadata = build_tdig_from_events(
        interactions,
        split_col=split_col,
        train_split_label=train_split_label,
        strict_before_timestamp=strict_before_timestamp,
        include_same_timestamp_transitions=include_same_timestamp_transitions,
    )
    dataset_provenance = _dataset_provenance(dataset_root)
    metadata.update(
        {
            "dataset_dir": str(dataset_root),
            "dataset_provenance": dataset_provenance,
            "split_name": split_name,
            "split_column": split_col,
            "train_split_label": train_split_label,
            "source_file": str(interactions_path),
        }
    )
    return graph, metadata


def build_tdig_from_events(
    events: pd.DataFrame,
    *,
    split_col: str | None = None,
    train_split_label: str = "train",
    strict_before_timestamp: int | None = None,
    include_same_timestamp_transitions: bool = False,
) -> tuple[TemporalDirectedItemGraph, dict[str, Any]]:
    """Build a TDIG from event rows.

    If ``split_col`` is provided, only rows equal to ``train_split_label`` are used.
    If ``strict_before_timestamp`` is provided, events at or after that timestamp are
    excluded. Together these options support leakage-safe train/as-of graph builds.
    """

    _validate_event_columns(events, split_col=split_col)
    frame = events.copy()
    if split_col is not None:
        frame = frame.loc[frame[split_col] == train_split_label].copy()
    if strict_before_timestamp is not None:
        frame = frame.loc[frame[schema.TIMESTAMP] < strict_before_timestamp].copy()
    frame = _with_event_id(frame)
    frame = frame.sort_values(
        [schema.USER_ID, schema.TIMESTAMP, schema.EVENT_ID, schema.ITEM_ID],
        kind="mergesort",
    ).reset_index(drop=True)

    edge_observations: dict[tuple[int, int], dict[str, Any]] = {}
    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    total_transitions = 0
    skipped_same_timestamp_transitions = 0

    for _, group in frame.groupby(schema.USER_ID, sort=True):
        ordered = group.sort_values(
            [schema.TIMESTAMP, schema.EVENT_ID, schema.ITEM_ID], kind="mergesort"
        )
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            if previous is not None:
                source_item_id = int(previous[schema.ITEM_ID])
                target_item_id = int(row[schema.ITEM_ID])
                source_timestamp = int(previous[schema.TIMESTAMP])
                target_timestamp = int(row[schema.TIMESTAMP])
                if (
                    not include_same_timestamp_transitions
                    and target_timestamp == source_timestamp
                ):
                    skipped_same_timestamp_transitions += 1
                    previous = row
                    continue
                gap_seconds = max(0, target_timestamp - source_timestamp)
                key = (source_item_id, target_item_id)
                observation = edge_observations.setdefault(
                    key,
                    {
                        "support": 0,
                        "timestamps": [],
                        "gaps": [],
                        "gap_histogram": Counter(),
                    },
                )
                observation["support"] += 1
                observation["timestamps"].append(target_timestamp)
                observation["gaps"].append(gap_seconds)
                observation["gap_histogram"][gap_bucket(gap_seconds)] += 1
                source_counts[source_item_id] += 1
                target_counts[target_item_id] += 1
                total_transitions += 1
            previous = row

    edges = _finalize_edges(
        edge_observations,
        source_counts=source_counts,
        target_counts=target_counts,
        total_transitions=total_transitions,
    )
    graph = TemporalDirectedItemGraph(edges)
    metadata = {
        "builder": "tdig_direct_transitions",
        "gap_bucket_boundaries_seconds": {
            "same_session_max": SAME_SESSION_SECONDS,
            "within_1d_max": DAY_SECONDS,
            "within_1w_max": WEEK_SECONDS,
            "within_1m_max": MONTH_SECONDS,
        },
        "input_event_count": int(len(events)),
        "used_event_count": int(len(frame)),
        "num_users": int(frame[schema.USER_ID].nunique()) if not frame.empty else 0,
        "num_source_items": int(len(source_counts)),
        "num_target_items": int(len(target_counts)),
        "num_edges": int(len(edges)),
        "num_transitions": int(total_transitions),
        "include_same_timestamp_transitions": include_same_timestamp_transitions,
        "skipped_same_timestamp_transitions": int(skipped_same_timestamp_transitions),
        "edge_stats": [
            "support",
            "transition_probability",
            "lift",
            "pmi",
            "direction_asymmetry",
            "first_transition_timestamp",
            "last_transition_timestamp",
            "mean_transition_timestamp",
            "mean_gap_seconds",
            "gap_histogram",
        ],
        "leakage_policy": (
            "Only events matching the requested training split are used; optional "
            "strict_before_timestamp excludes events at or after a prediction time."
        ),
    }
    if split_col is not None:
        metadata["split_column"] = split_col
        metadata["train_split_label"] = train_split_label
    if strict_before_timestamp is not None:
        metadata["strict_before_timestamp"] = int(strict_before_timestamp)
    return graph, metadata


def build_tdig_artifact(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    split_name: str = "temporal_leave_one_out",
    train_split_label: str = "train",
    strict_before_timestamp: int | None = None,
    include_same_timestamp_transitions: bool = False,
    command: str = "tglrec graph build-tdig",
) -> TDIGBuildResult:
    """Build a train-only TDIG artifact under ``output_dir``."""

    graph, metadata = build_tdig_from_processed_split(
        dataset_dir=dataset_dir,
        split_name=split_name,
        train_split_label=train_split_label,
        strict_before_timestamp=strict_before_timestamp,
        include_same_timestamp_transitions=include_same_timestamp_transitions,
    )
    root = ensure_dir(output_dir)
    _write_edges_csv(graph.to_edge_rows(), root / "edges.csv")
    config = {
        "builder": "tdig_direct_transitions",
        "dataset_dir": str(Path(dataset_dir)),
        "split_name": split_name,
        "train_split_label": train_split_label,
        "strict_before_timestamp": strict_before_timestamp,
        "include_same_timestamp_transitions": include_same_timestamp_transitions,
    }
    artifact_metadata = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "edges": "edges.csv",
            "metadata": "metadata.json",
            "config": "config.yaml",
            "checksums": "checksums.json",
        },
    }
    write_artifact_manifest(root, command=command, config=config, metadata=artifact_metadata)
    _write_environment(root / "environment.json")
    (root / "stdout.log").write_text(
        json.dumps(
            {
                "output_dir": str(root),
                "num_edges": metadata["num_edges"],
                "num_transitions": metadata["num_transitions"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "stderr.log").write_text("", encoding="utf-8", newline="\n")
    write_checksum_manifest(root, TDIG_ARTIFACT_FILES)
    return TDIGBuildResult(
        output_dir=root,
        num_edges=metadata["num_edges"],
        num_transitions=metadata["num_transitions"],
        metadata=artifact_metadata,
    )


def _finalize_edges(
    edge_observations: dict[tuple[int, int], dict[str, Any]],
    *,
    source_counts: Counter[int],
    target_counts: Counter[int],
    total_transitions: int,
) -> list[EdgeStats]:
    edges: list[EdgeStats] = []
    for (source_item_id, target_item_id), observation in sorted(edge_observations.items()):
        support = int(observation["support"])
        source_transition_count = int(source_counts[source_item_id])
        target_transition_count = int(target_counts[target_item_id])
        transition_probability = support / source_transition_count
        target_probability = target_transition_count / total_transitions if total_transitions else 0.0
        lift = transition_probability / target_probability if target_probability > 0 else 0.0
        pmi = math.log(lift) if lift > 0 else 0.0
        reverse_support = int(edge_observations.get((target_item_id, source_item_id), {}).get("support", 0))
        direction_asymmetry = (
            (support - reverse_support) / (support + reverse_support)
            if support + reverse_support > 0
            else 0.0
        )
        timestamps = [int(value) for value in observation["timestamps"]]
        gaps = [int(value) for value in observation["gaps"]]
        gap_histogram = {
            bucket: int(observation["gap_histogram"].get(bucket, 0)) for bucket in GAP_BUCKETS
        }
        edges.append(
            EdgeStats(
                source_item_id=source_item_id,
                target_item_id=target_item_id,
                support=support,
                source_transition_count=source_transition_count,
                target_transition_count=target_transition_count,
                total_transition_count=total_transitions,
                transition_probability=transition_probability,
                lift=lift,
                pmi=pmi,
                direction_asymmetry=direction_asymmetry,
                first_transition_timestamp=min(timestamps),
                last_transition_timestamp=max(timestamps),
                mean_transition_timestamp=sum(timestamps) / len(timestamps),
                mean_gap_seconds=sum(gaps) / len(gaps),
                gap_histogram=gap_histogram,
            )
        )
    return edges


def _edge_score(edge: EdgeStats, *, score_field: str, gap_bucket: str | None) -> float:
    if gap_bucket is not None:
        return float(edge.gap_histogram.get(gap_bucket, 0))
    if not hasattr(edge, score_field):
        raise ValueError(f"Unknown edge score field: {score_field}")
    value = getattr(edge, score_field)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Edge score field must be numeric: {score_field}")
    return float(value)


def _edge_identity_key(edge: EdgeStats) -> tuple[int, int]:
    return (edge.source_item_id, edge.target_item_id)


def _split_column(split_name: str) -> str:
    if split_name == "temporal_leave_one_out":
        return schema.SPLIT_LOO
    if split_name == "global_time":
        return schema.SPLIT_GLOBAL
    raise ValueError("split_name must be 'temporal_leave_one_out' or 'global_time'")


def _dataset_provenance(dataset_root: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "interactions_csv": file_fingerprint(dataset_root / "interactions.csv", include_path=True),
    }
    optional_files = {
        "dataset_checksums": dataset_root / CHECKSUM_MANIFEST_NAME,
        "dataset_config": dataset_root / "config.yaml",
        "dataset_metadata": dataset_root / "metadata.json",
    }
    missing: list[str] = []
    for key, path in optional_files.items():
        if path.is_file():
            provenance[key] = file_fingerprint(path, include_path=True)
        else:
            missing.append(path.name)
    if (dataset_root / CHECKSUM_MANIFEST_NAME).is_file():
        manifest = read_json(dataset_root / CHECKSUM_MANIFEST_NAME)
        interactions = manifest.get("files", {}).get("interactions.csv")
        if isinstance(interactions, dict):
            provenance["interactions_csv_from_dataset_manifest"] = interactions
    if missing:
        provenance["warnings"] = [
            "Missing processed dataset provenance files: " + ", ".join(sorted(missing))
        ]
    return provenance


def _validate_event_columns(events: pd.DataFrame, *, split_col: str | None) -> None:
    required = {schema.USER_ID, schema.ITEM_ID, schema.TIMESTAMP}
    if split_col is not None:
        required.add(split_col)
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Missing required TDIG event columns: {missing}")


def _with_event_id(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    if schema.EVENT_ID in frame.columns:
        frame[schema.EVENT_ID] = frame[schema.EVENT_ID].astype("int64")
        return frame
    tied_times = frame.duplicated([schema.USER_ID, schema.TIMESTAMP], keep=False)
    if tied_times.any():
        example = frame.loc[tied_times, [schema.USER_ID, schema.TIMESTAMP]].iloc[0].to_dict()
        raise ValueError(
            "TDIG event rows without event_id must not contain same-user same-timestamp ties; "
            f"provide deterministic event_id values or resolve ties before graph construction. "
            f"Example ambiguous key: {example}"
        )
    frame = frame.reset_index(drop=True)
    frame[schema.EVENT_ID] = range(len(frame))
    return frame


def _write_edges_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_item_id",
        "target_item_id",
        "support",
        "source_transition_count",
        "target_transition_count",
        "total_transition_count",
        "transition_probability",
        "lift",
        "pmi",
        "direction_asymmetry",
        "first_transition_timestamp",
        "last_transition_timestamp",
        "mean_transition_timestamp",
        "mean_gap_seconds",
        *(f"gap_{bucket}" for bucket in GAP_BUCKETS),
        "gap_histogram_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_environment(path: Path) -> None:
    write_json(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "git_commit": current_git_commit("."),
            "package_versions": _package_versions(["tglrec", "pandas", "numpy", "pyyaml"]),
        },
        path,
    )


def _package_versions(package_names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions
