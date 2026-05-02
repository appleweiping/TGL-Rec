"""Configuration and safeguards for vectorized shared-pool scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path


_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


@dataclass(frozen=True)
class SharedPoolScoringConfig:
    """Runtime controls for compact shared-pool scoring."""

    candidate_output_mode: str = "compact_ref"
    batch_size: int = 512
    top_n_to_save: int = 100
    max_candidate_size: int = 1000
    device: str = "auto"
    score_dtype: str = "float32"
    verify_candidate_checksum: bool = True
    tie_break: str = "item_id_ascending"
    save_full_scores: bool = False
    save_top_evidence_only: bool = True
    max_batch_candidate_scores_memory_mb: float = 512.0
    max_prediction_file_size_gb_warning: float = 2.0
    flush_every_n_rows: int = 1000
    progress_every_n_batches: int = 20
    max_k: int = 10

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "SharedPoolScoringConfig":
        """Build config from a mapping, ignoring unknown fields for forward compatibility."""

        payload = dict(payload or {})
        allowed = {field for field in cls.__dataclass_fields__}
        values = {key: value for key, value in payload.items() if key in allowed}
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SharedPoolScoringConfig":
        """Load scoring controls from YAML."""

        return cls.from_dict(load_yaml_config(resolve_path(path)))

    def validate(self) -> None:
        """Validate non-negotiable compact scoring constraints."""

        if self.candidate_output_mode not in {"compact_ref", "compact_ref_v1"}:
            raise ValueError("shared-pool scoring must use compact_ref output")
        if int(self.top_n_to_save) < int(self.max_k):
            raise ValueError("top_n_to_save must be >= max ranking K")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.max_candidate_size) <= 0:
            raise ValueError("max_candidate_size must be positive")
        if self.score_dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported score_dtype: {self.score_dtype}")
        if self.tie_break != "item_id_ascending":
            raise ValueError("only deterministic item_id_ascending tie-break is supported")
        if bool(self.save_full_scores):
            raise ValueError("save_full_scores=true would expand all shared-pool scores")


def load_shared_pool_scoring_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SharedPoolScoringConfig:
    """Load shared-pool config with optional call-site overrides."""

    payload: dict[str, Any] = {}
    if path is not None and resolve_path(path).is_file():
        payload.update(load_yaml_config(resolve_path(path)))
    if overrides:
        payload.update(overrides)
    return SharedPoolScoringConfig.from_dict(payload)


def estimate_score_tensor_memory_mb(
    *,
    batch_size: int,
    candidate_size: int,
    score_dtype: str = "float32",
) -> float:
    """Estimate score tensor memory in MiB."""

    bytes_per_value = _DTYPE_BYTES.get(str(score_dtype), 4)
    return int(batch_size) * int(candidate_size) * bytes_per_value / float(1024 * 1024)


def adjusted_batch_size(
    *,
    requested_batch_size: int,
    candidate_size: int,
    score_dtype: str,
    max_memory_mb: float,
) -> int:
    """Return a safe batch size, reducing automatically when score memory is too high."""

    requested = int(requested_batch_size)
    candidates = int(candidate_size)
    if candidates <= 0:
        raise ValueError("candidate_size must be positive")
    if estimate_score_tensor_memory_mb(
        batch_size=requested,
        candidate_size=candidates,
        score_dtype=score_dtype,
    ) <= float(max_memory_mb):
        return requested
    bytes_per_value = _DTYPE_BYTES.get(str(score_dtype), 4)
    max_values = int(float(max_memory_mb) * 1024 * 1024 / bytes_per_value)
    safe = max_values // candidates
    if safe < 1:
        raise MemoryError(
            "score tensor memory limit is too small for one shared-pool row: "
            f"candidate_size={candidates} dtype={score_dtype} limit_mb={max_memory_mb}"
        )
    return max(1, min(requested, safe))
