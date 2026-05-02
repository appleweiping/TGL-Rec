"""Atomic compact top-k prediction writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from llm4rec.evaluation.prediction_schema import CANDIDATE_SCHEMA_COMPACT_REF, PREDICTION_SCHEMA_V2
from llm4rec.scoring.candidate_batch import CandidateBatch
from llm4rec.scoring.topk import top_k_items_and_scores


RowMetadataFn = Callable[[int, list[str], list[float]], dict[str, Any]]


class CompactTopKPredictionWriter:
    """Write compact_ref_v1 prediction rows through a temp file and atomic rename."""

    def __init__(
        self,
        path: str | Path,
        *,
        method: str,
        top_n_to_save: int,
        flush_every_n_rows: int = 1000,
        base_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        self.method = str(method)
        self.top_n_to_save = int(top_n_to_save)
        self.flush_every_n_rows = int(flush_every_n_rows)
        self.base_metadata = dict(base_metadata or {})
        self._handle: Any | None = None
        self.rows_written = 0
        if self.top_n_to_save <= 0:
            raise ValueError("top_n_to_save must be positive")

    def __enter__(self) -> "CompactTopKPredictionWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self._handle = self.tmp_path.open("w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        elif self.tmp_path.exists():
            self.tmp_path.unlink()

    def write_batch(
        self,
        *,
        batch: CandidateBatch,
        score_matrix: Any,
        scorer_name: str,
        row_metadata_fn: RowMetadataFn | None = None,
    ) -> list[dict[str, Any]]:
        """Write one scored batch and return the compact rows for metrics."""

        if self._handle is None:
            raise RuntimeError("writer is not open")
        output_rows: list[dict[str, Any]] = []
        for index, (row_scores, item_ids) in enumerate(zip(score_matrix, batch.candidate_item_ids)):
            predicted_items, scores = top_k_items_and_scores(
                row_scores,
                item_ids,
                top_n=self.top_n_to_save,
            )
            metadata = {
                **self.base_metadata,
                "candidate_schema": CANDIDATE_SCHEMA_COMPACT_REF,
                "scorer": str(scorer_name),
                "top_n_to_save": self.top_n_to_save,
            }
            if row_metadata_fn is not None:
                metadata.update(row_metadata_fn(index, predicted_items, scores))
            row = {
                "candidate_ref": batch.candidate_refs[index],
                "domain": batch.domains[index],
                "metadata": metadata,
                "method": self.method,
                "predicted_items": predicted_items,
                "raw_output": None,
                "schema_version": PREDICTION_SCHEMA_V2,
                "scores": scores,
                "target_item": batch.target_items[index],
                "user_id": batch.user_ids[index],
            }
            self._handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            output_rows.append(row)
            self.rows_written += 1
            if self.flush_every_n_rows > 0 and self.rows_written % self.flush_every_n_rows == 0:
                self._handle.flush()
        return output_rows
