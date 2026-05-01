"""Prompt data contracts for Phase 3A LLM diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PromptBuildError(ValueError):
    """Raised when a diagnostic prompt cannot be constructed."""


@dataclass(frozen=True)
class PromptExample:
    """One fixed candidate reranking case for sequence/time prompt diagnostics."""

    user_id: str
    history: list[str]
    target_item: str
    candidate_items: list[str]
    item_records: dict[str, dict[str, Any]]
    domain: str | None = None
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    transition_evidence: list[dict[str, Any]] = field(default_factory=list)
    time_window_evidence: list[dict[str, Any]] = field(default_factory=list)
    contrastive_evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptRequest:
    """A rendered prompt plus audit metadata."""

    prompt: str
    prompt_variant: str
    prompt_version: str
    candidate_item_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

