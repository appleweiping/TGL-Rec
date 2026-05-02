"""Candidate resolution for expanded and compact prediction rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.data.candidates import candidate_row_id
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import sha256_file, write_json


class CandidateResolutionError(ValueError):
    """Raised when compact candidate references cannot be resolved safely."""


class CandidateResolver:
    """Resolve frozen candidates for prediction validation.

    Compact references avoid repeating large candidate lists in prediction JSONL.
    The resolver verifies artifact checksums and reconstructs the exact candidate
    set from the frozen candidate artifact or its shared-pool sidecar.
    """

    _verified_sha: dict[tuple[str, str], bool] = {}

    def __init__(
        self,
        *,
        candidate_artifact_path: str | Path,
        candidate_artifact_sha256: str,
        candidate_pool_path: str | Path | None = None,
        candidate_pool_sha256: str | None = None,
    ) -> None:
        self.candidate_artifact_path = resolve_path(candidate_artifact_path)
        self.candidate_artifact_sha256 = str(candidate_artifact_sha256).lower()
        self.candidate_pool_path = resolve_path(candidate_pool_path) if candidate_pool_path else self._default_pool_path()
        self.candidate_pool_sha256 = None if candidate_pool_sha256 is None else str(candidate_pool_sha256).lower()
        self._candidate_pool: dict[str, Any] | None = None
        self._offset_index: dict[str, int] | None = None
        self.verify_checksum()

    @classmethod
    def from_ref(cls, candidate_ref: dict[str, Any]) -> "CandidateResolver":
        """Create a resolver from a prediction row candidate_ref."""

        return cls(
            candidate_artifact_path=candidate_ref["artifact_path"],
            candidate_artifact_sha256=candidate_ref["artifact_sha256"],
            candidate_pool_path=candidate_ref.get("candidate_pool_artifact"),
            candidate_pool_sha256=candidate_ref.get("candidate_pool_sha256"),
        )

    def verify_checksum(self) -> None:
        """Verify the frozen candidate artifact and optional pool artifact."""

        key = (str(self.candidate_artifact_path), self.candidate_artifact_sha256)
        if key not in self._verified_sha:
            actual = sha256_file(self.candidate_artifact_path)
            if actual.lower() != self.candidate_artifact_sha256:
                raise CandidateResolutionError(
                    "candidate artifact checksum mismatch: "
                    f"expected={self.candidate_artifact_sha256} actual={actual}"
                )
            self._verified_sha[key] = True
        if self.candidate_pool_path and self.candidate_pool_path.is_file() and self.candidate_pool_sha256:
            pool_key = (str(self.candidate_pool_path), self.candidate_pool_sha256)
            if pool_key not in self._verified_sha:
                actual = sha256_file(self.candidate_pool_path)
                if actual.lower() != self.candidate_pool_sha256:
                    raise CandidateResolutionError(
                        "candidate pool checksum mismatch: "
                        f"expected={self.candidate_pool_sha256} actual={actual}"
                    )
                self._verified_sha[pool_key] = True

    def resolve_prediction_row(self, row: dict[str, Any]) -> list[str]:
        """Resolve candidates for an expanded or compact prediction row."""

        candidate_items = row.get("candidate_items")
        if isinstance(candidate_items, list) and candidate_items:
            return [str(item) for item in candidate_items]
        candidate_ref = row.get("candidate_ref")
        if not isinstance(candidate_ref, dict):
            raise CandidateResolutionError("prediction row has no candidate_items or candidate_ref")
        return self.get_candidates(
            candidate_row_id_value=str(candidate_ref["candidate_row_id"]),
            user_id=str(row.get("user_id", "")),
            target_item=str(row["target_item"]),
            candidate_ref=candidate_ref,
        )

    def get_candidates(
        self,
        *,
        candidate_row_id_value: str | None = None,
        user_id: str | None = None,
        target_item: str,
        split: str = "test",
        candidate_ref: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return the frozen candidate list and verify target inclusion."""

        ref = dict(candidate_ref or {})
        storage = str(ref.get("candidate_storage", ""))
        if storage == "shared_pool" or self._has_pool():
            candidates = self._shared_pool_candidates(str(target_item), ref)
        else:
            row = self._candidate_row(
                candidate_row_id_value
                or candidate_row_id(user_id=str(user_id), target_item=str(target_item), split=split)
            )
            candidates = [str(item) for item in row.get("candidate_items", [])]
        if str(target_item) not in candidates:
            raise CandidateResolutionError(f"target_item missing from resolved candidates: {target_item}")
        expected_size = ref.get("candidate_size")
        if expected_size is not None and len(candidates) != int(expected_size):
            raise CandidateResolutionError(
                f"candidate size mismatch: expected={expected_size} actual={len(candidates)}"
            )
        return candidates

    @property
    def resolver_mode(self) -> str:
        """Return the resolver mode used for metadata."""

        return "shared_pool" if self._has_pool() else "offset_index"

    def _default_pool_path(self) -> Path | None:
        pool = self.candidate_artifact_path.parent / "candidate_pool.json"
        return pool if pool.is_file() else None

    def _has_pool(self) -> bool:
        return bool(self.candidate_pool_path and self.candidate_pool_path.is_file())

    def _load_pool(self) -> dict[str, Any]:
        if not self._has_pool():
            raise CandidateResolutionError("candidate pool artifact is unavailable")
        if self._candidate_pool is None:
            assert self.candidate_pool_path is not None
            self._candidate_pool = json.loads(self.candidate_pool_path.read_text(encoding="utf-8"))
        return self._candidate_pool

    def _shared_pool_candidates(self, target_item: str, candidate_ref: dict[str, Any]) -> list[str]:
        pool_payload = self._load_pool()
        pool = [str(item) for item in pool_payload.get("candidate_items", [])]
        if target_item in pool:
            return pool
        negatives = [
            str(item)
            for item in pool_payload.get("negative_pool_for_targets_outside_pool", pool[:-1])
        ]
        candidates = [*negatives, target_item]
        expected_size = int(candidate_ref.get("candidate_size", pool_payload.get("candidate_size", len(pool))))
        if len(candidates) != expected_size:
            raise CandidateResolutionError(
                f"shared-pool candidate size mismatch: expected={expected_size} actual={len(candidates)}"
            )
        return candidates

    def _candidate_row(self, row_id: str) -> dict[str, Any]:
        offsets = self._load_or_build_offset_index()
        if row_id not in offsets:
            raise CandidateResolutionError(f"candidate_row_id not found: {row_id}")
        with self.candidate_artifact_path.open("rb") as handle:
            handle.seek(int(offsets[row_id]))
            line = handle.readline().decode("utf-8")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise CandidateResolutionError(f"candidate row is not an object: {row_id}")
        return row

    def _load_or_build_offset_index(self) -> dict[str, int]:
        if self._offset_index is not None:
            return self._offset_index
        index_path = self.candidate_artifact_path.parent / "candidate_index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            if payload.get("source_sha256") == self.candidate_artifact_sha256:
                self._offset_index = {str(key): int(value) for key, value in payload.get("offsets", {}).items()}
                return self._offset_index
        offsets: dict[str, int] = {}
        with self.candidate_artifact_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line.decode("utf-8"))
                row_id = str(
                    row.get("candidate_row_id")
                    or candidate_row_id(
                        user_id=str(row["user_id"]),
                        target_item=str(row["target_item"]),
                        split=str(row.get("split", "test")),
                    )
                )
                offsets[row_id] = offset
        write_json(
            index_path,
            {
                "index_type": "byte_offset",
                "offsets": offsets,
                "source_artifact": str(self.candidate_artifact_path),
                "source_sha256": self.candidate_artifact_sha256,
            },
        )
        self._offset_index = offsets
        return offsets
