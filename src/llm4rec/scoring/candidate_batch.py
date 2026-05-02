"""Streaming batches over frozen compact candidate references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.data.candidates import candidate_row_id
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import sha256_file


@dataclass(frozen=True)
class CandidateBatch:
    """One vectorized scoring batch over resolved candidate item IDs."""

    user_ids: list[str]
    histories: list[list[str]]
    target_items: list[str]
    candidate_item_ids: list[list[str]]
    domains: list[str | None]
    candidate_refs: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]
    prediction_timestamps: list[float | None]

    @property
    def batch_size(self) -> int:
        return len(self.user_ids)

    @property
    def candidate_size(self) -> int:
        return len(self.candidate_item_ids[0]) if self.candidate_item_ids else 0


class CandidateBatchIterator:
    """Lazily read frozen candidate rows and emit resolved scoring batches."""

    _verified_sha: set[tuple[str, str]] = set()

    def __init__(
        self,
        *,
        candidate_artifact_path: str | Path,
        candidate_artifact_sha256: str,
        history_by_user: dict[str, list[str]],
        batch_size: int,
        candidate_pool_path: str | Path | None = None,
        candidate_pool_sha256: str | None = None,
        timestamp_by_user: dict[str, float | None] | None = None,
        split: str = "test",
        verify_candidate_checksum: bool = True,
        artifact_id: str | None = None,
    ) -> None:
        self.candidate_artifact_path = resolve_path(candidate_artifact_path)
        self.candidate_artifact_sha256 = str(candidate_artifact_sha256).lower()
        self.candidate_pool_path = resolve_path(candidate_pool_path) if candidate_pool_path else self._default_pool_path()
        self.candidate_pool_sha256 = None if candidate_pool_sha256 is None else str(candidate_pool_sha256).lower()
        self.history_by_user = {str(user): [str(item) for item in items] for user, items in history_by_user.items()}
        self.timestamp_by_user = {
            str(user): (None if value is None else float(value))
            for user, value in dict(timestamp_by_user or {}).items()
        }
        self.batch_size = int(batch_size)
        self.split = str(split)
        self.artifact_id = artifact_id or f"{self.candidate_artifact_path.parent.name}_candidates_protocol_v1"
        self._candidate_pool: dict[str, Any] | None = None
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if verify_candidate_checksum:
            self.verify_checksum()

    def verify_checksum(self) -> None:
        """Verify candidate and pool checksums once per process."""

        self._verify_one(self.candidate_artifact_path, self.candidate_artifact_sha256, "candidate artifact")
        if self.candidate_pool_path is not None and self.candidate_pool_path.is_file() and self.candidate_pool_sha256:
            self._verify_one(self.candidate_pool_path, self.candidate_pool_sha256, "candidate pool")

    def __iter__(self) -> Iterable[CandidateBatch]:
        users: list[str] = []
        histories: list[list[str]] = []
        targets: list[str] = []
        candidates: list[list[str]] = []
        domains: list[str | None] = []
        refs: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        timestamps: list[float | None] = []
        with self.candidate_artifact_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("split")) != self.split:
                    continue
                user_id = str(row["user_id"])
                target = str(row["target_item"])
                resolved = self.resolve_candidates(row)
                users.append(user_id)
                histories.append([item for item in self.history_by_user.get(user_id, []) if item != target])
                targets.append(target)
                candidates.append(resolved)
                domains.append(None if row.get("domain") is None else str(row.get("domain")))
                refs.append(self.candidate_ref(row))
                rows.append(row)
                timestamps.append(self.timestamp_by_user.get(user_id))
                if len(users) >= self.batch_size:
                    yield CandidateBatch(users, histories, targets, candidates, domains, refs, rows, timestamps)
                    users, histories, targets, candidates, domains, refs, rows, timestamps = [], [], [], [], [], [], [], []
        if users:
            yield CandidateBatch(users, histories, targets, candidates, domains, refs, rows, timestamps)

    def resolve_candidates(self, row: dict[str, Any]) -> list[str]:
        """Resolve row candidates from shared-pool or expanded storage."""

        target = str(row["target_item"])
        if str(row.get("candidate_storage", "")) == "shared_pool" or self._has_pool():
            pool = self._load_pool()
            pool_items = pool["_candidate_items_list"]
            if target in pool["_candidate_items_set"]:
                candidates = pool_items
            else:
                candidates = [*pool["_negative_pool_list"], target]
        else:
            candidates = [str(item) for item in row.get("candidate_items", [])]
        expected_size = int(row.get("candidate_size", len(candidates)))
        if len(candidates) != expected_size:
            raise ValueError(f"candidate size mismatch: expected={expected_size} actual={len(candidates)}")
        if target not in candidates:
            raise ValueError(f"target_item missing from resolved candidates: {target}")
        return candidates

    def candidate_ref(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build compact_ref_v1 metadata for a candidate row."""

        row_id = str(
            row.get("candidate_row_id")
            or candidate_row_id(
                user_id=str(row["user_id"]),
                target_item=str(row["target_item"]),
                split=str(row.get("split", self.split)),
            )
        )
        ref: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "artifact_path": str(self.candidate_artifact_path),
            "artifact_sha256": self.candidate_artifact_sha256,
            "candidate_row_id": row_id,
            "candidate_size": int(row.get("candidate_size", len(self.resolve_candidates(row)))),
            "candidate_storage": str(row.get("candidate_storage", "expanded")),
            "split": str(row.get("split", self.split)),
        }
        if self.candidate_pool_path is not None:
            ref["candidate_pool_artifact"] = str(self.candidate_pool_path)
        if self.candidate_pool_sha256:
            ref["candidate_pool_sha256"] = self.candidate_pool_sha256
        if row.get("target_inclusion_rule"):
            ref["target_inclusion_rule"] = str(row["target_inclusion_rule"])
        return ref

    def _default_pool_path(self) -> Path | None:
        pool = self.candidate_artifact_path.parent / "candidate_pool.json"
        return pool if pool.is_file() else None

    def _has_pool(self) -> bool:
        return bool(self.candidate_pool_path and self.candidate_pool_path.is_file())

    def _load_pool(self) -> dict[str, Any]:
        if not self._has_pool():
            raise FileNotFoundError(f"missing candidate pool for {self.candidate_artifact_path}")
        if self._candidate_pool is None:
            assert self.candidate_pool_path is not None
            payload = json.loads(self.candidate_pool_path.read_text(encoding="utf-8"))
            pool_items = [str(item) for item in payload.get("candidate_items", [])]
            negative_pool = [
                str(item)
                for item in payload.get("negative_pool_for_targets_outside_pool", pool_items[:-1])
            ]
            payload["_candidate_items_list"] = pool_items
            payload["_candidate_items_set"] = set(pool_items)
            payload["_negative_pool_list"] = negative_pool
            self._candidate_pool = payload
        return self._candidate_pool

    def _verify_one(self, path: Path, expected_sha256: str, label: str) -> None:
        key = (str(path), str(expected_sha256).lower())
        if key in self._verified_sha:
            return
        actual = sha256_file(path).lower()
        if actual != str(expected_sha256).lower():
            raise ValueError(f"{label} checksum mismatch: expected={expected_sha256} actual={actual}")
        self._verified_sha.add(key)
