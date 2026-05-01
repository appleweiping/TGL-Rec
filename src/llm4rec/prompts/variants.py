"""Prompt variant definitions for sequence/time sensitivity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVariantSpec:
    name: str
    prompt_version: str
    uses_order: bool
    uses_time_gaps: bool
    uses_time_buckets: bool
    uses_transition_evidence: bool
    uses_time_window_evidence: bool
    uses_contrastive_evidence: bool


PROMPT_VARIANTS: dict[str, PromptVariantSpec] = {
    "history_only": PromptVariantSpec(
        name="history_only",
        prompt_version="phase3a.history_only.v1",
        uses_order=False,
        uses_time_gaps=False,
        uses_time_buckets=False,
        uses_transition_evidence=False,
        uses_time_window_evidence=False,
        uses_contrastive_evidence=False,
    ),
    "history_with_order": PromptVariantSpec(
        name="history_with_order",
        prompt_version="phase3a.history_with_order.v1",
        uses_order=True,
        uses_time_gaps=False,
        uses_time_buckets=False,
        uses_transition_evidence=False,
        uses_time_window_evidence=False,
        uses_contrastive_evidence=False,
    ),
    "history_with_time_gaps": PromptVariantSpec(
        name="history_with_time_gaps",
        prompt_version="phase3a.history_with_time_gaps.v1",
        uses_order=True,
        uses_time_gaps=True,
        uses_time_buckets=False,
        uses_transition_evidence=False,
        uses_time_window_evidence=False,
        uses_contrastive_evidence=False,
    ),
    "history_with_time_buckets": PromptVariantSpec(
        name="history_with_time_buckets",
        prompt_version="phase3a.history_with_time_buckets.v1",
        uses_order=True,
        uses_time_gaps=False,
        uses_time_buckets=True,
        uses_transition_evidence=False,
        uses_time_window_evidence=False,
        uses_contrastive_evidence=False,
    ),
    "history_with_transition_evidence": PromptVariantSpec(
        name="history_with_transition_evidence",
        prompt_version="phase3a.history_with_transition_evidence.v1",
        uses_order=True,
        uses_time_gaps=False,
        uses_time_buckets=True,
        uses_transition_evidence=True,
        uses_time_window_evidence=False,
        uses_contrastive_evidence=False,
    ),
    "history_with_time_window_evidence": PromptVariantSpec(
        name="history_with_time_window_evidence",
        prompt_version="phase3a.history_with_time_window_evidence.v1",
        uses_order=True,
        uses_time_gaps=False,
        uses_time_buckets=True,
        uses_transition_evidence=False,
        uses_time_window_evidence=True,
        uses_contrastive_evidence=False,
    ),
    "history_with_contrastive_evidence": PromptVariantSpec(
        name="history_with_contrastive_evidence",
        prompt_version="phase3a.history_with_contrastive_evidence.v1",
        uses_order=True,
        uses_time_gaps=False,
        uses_time_buckets=True,
        uses_transition_evidence=True,
        uses_time_window_evidence=True,
        uses_contrastive_evidence=True,
    ),
}


def get_prompt_variant(name: str) -> PromptVariantSpec:
    """Return a registered prompt variant specification."""

    key = str(name)
    if key not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {name}")
    return PROMPT_VARIANTS[key]


def prompt_variant_names() -> list[str]:
    """Return deterministic prompt variant names."""

    return sorted(PROMPT_VARIANTS)

