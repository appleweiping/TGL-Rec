"""SFT variant registry for local LoRA recommendation baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTVariantSpec:
    """How one LoRA SFT baseline conditions the shared reranking task."""

    name: str
    family: str
    description: str
    evidence_lines: tuple[str, ...] = ()
    reference_group: str | None = None
    observation_axes: tuple[str, ...] = ()

    def evidence_block(self) -> str:
        """Return newline-prefixed evidence text for the user prompt."""

        if not self.evidence_lines:
            return ""
        return "\n" + "\n".join(self.evidence_lines)


SFT_VARIANT_REGISTRY: dict[str, SFTVariantSpec] = {
    "history_only_sft": SFTVariantSpec(
        name="history_only_sft",
        family="control",
        description="ID-history-only LoRA reranker with no explicit temporal or observation evidence.",
        observation_axes=("history_order",),
    ),
    "temporal_evidence_sft": SFTVariantSpec(
        name="temporal_evidence_sft",
        family="ours_observation",
        description="LoRA reranker conditioned on temporal transition and contrastive evidence.",
        evidence_lines=(
            "Time buckets: recent history items are later in the sequence.",
            "Transition evidence: rank candidates likely to follow the recent history.",
            "Contrastive evidence: distinguish semantic similarity from next-need transitions.",
        ),
        observation_axes=("history_order", "temporal_bucket", "transition", "contrastive"),
    ),
    "reference_preference_sft": SFTVariantSpec(
        name="reference_preference_sft",
        family="reference_baseline",
        description=(
            "Reference-style preference-conditioned LoRA reranker; use for papers whose core "
            "signal is user preference alignment, controllability, or instruction preference."
        ),
        evidence_lines=(
            "Preference evidence: infer stable user preference from the interaction history.",
            "Control evidence: keep the ranking faithful to the provided candidate set and task constraints.",
        ),
        reference_group="references",
        observation_axes=("preference_alignment", "candidate_grounding"),
    ),
    "reference_semantic_sft": SFTVariantSpec(
        name="reference_semantic_sft",
        family="reference_baseline",
        description=(
            "Reference-style semantic matching LoRA reranker; use for papers whose core signal is "
            "textual, multimodal, or semantic item matching rather than temporal transitions."
        ),
        evidence_lines=(
            "Semantic evidence: rank candidates by preference-compatible semantic match to history items.",
            "Robustness evidence: avoid overfitting to item-ID frequency when semantic evidence conflicts.",
        ),
        reference_group="references",
        observation_axes=("semantic_similarity", "candidate_grounding"),
    ),
    "reference_long_tail_sft": SFTVariantSpec(
        name="reference_long_tail_sft",
        family="reference_baseline",
        description=(
            "Reference-style long-tail-aware LoRA reranker; use for papers focused on long-tail "
            "or popularity-bias mitigation under the shared evaluator."
        ),
        evidence_lines=(
            "Long-tail evidence: consider whether less frequent candidates fit the user's recent need.",
            "Popularity-bias check: do not rank an item higher solely because it is globally popular.",
        ),
        reference_group="references",
        observation_axes=("long_tail", "popularity_bias", "candidate_grounding"),
    ),
    "reference_collaborative_sft": SFTVariantSpec(
        name="reference_collaborative_sft",
        family="reference_baseline",
        description=(
            "Reference-style collaborative-signal LoRA reranker; use for papers or baselines whose "
            "core signal is item co-occurrence, user-neighborhood preference, or sequential "
            "collaborative filtering rather than explicit semantic evidence."
        ),
        evidence_lines=(
            "Collaborative evidence: rank candidates that frequently co-occur with the user's history.",
            "Neighborhood evidence: prefer candidates supported by similar users or adjacent sequences.",
            "Sequence-collaboration check: separate collaborative transition support from plain popularity.",
        ),
        reference_group="references",
        observation_axes=("collaborative_signal", "cooccurrence", "candidate_grounding", "popularity_bias"),
    ),
}


def get_sft_variant(name: str) -> SFTVariantSpec:
    """Return a registered SFT variant."""

    key = str(name)
    if key not in SFT_VARIANT_REGISTRY:
        supported = ", ".join(sorted(SFT_VARIANT_REGISTRY))
        raise ValueError(f"Unsupported SFT variant: {name}. Supported variants: {supported}")
    return SFT_VARIANT_REGISTRY[key]


def sft_variant_names() -> list[str]:
    """Return deterministic SFT variant names."""

    return sorted(SFT_VARIANT_REGISTRY)
