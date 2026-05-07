"""Concrete senior-reference methods selected for faithful adaptation."""

from __future__ import annotations

from dataclasses import dataclass


class ReferenceBaselineNotImplementedError(NotImplementedError):
    """Raised when a selected reference baseline has not been faithfully adapted."""


@dataclass(frozen=True)
class ReferenceMethodSpec:
    """Traceable plan for adapting one original reference method."""

    baseline_id: str
    title: str
    local_reference: str
    public_url: str
    official_code_url: str | None
    official_code_status: str
    method_family: str
    preserve_components: tuple[str, ...]
    fair_protocol_adaptations: tuple[str, ...]
    implementation_status: str = "not_implemented"
    reportable_baseline: bool = False


REFERENCE_METHOD_REGISTRY: dict[str, ReferenceMethodSpec] = {
    "slmrec_distill_qwen_lora": ReferenceMethodSpec(
        baseline_id="slmrec_distill_qwen_lora",
        title="SLMRec: Distilling Large Language Models into Small Recommendation Models",
        local_reference="references/NH/11465_SLMRec_Distilling_Large_.pdf",
        public_url="https://arxiv.org/abs/2405.17890",
        official_code_url="https://github.com/WujiangXu/SLMRec",
        official_code_status="official_code_identified",
        method_family="distillation_sequential_recommendation",
        preserve_components=(
            "depth/knowledge distillation objective",
            "teacher-student recommendation supervision",
            "sequential recommendation training signal",
        ),
        fair_protocol_adaptations=(
            "use Qwen3-8B LoRA/QLoRA where the reference method requires an LLM backbone",
            "use frozen TGL-Rec splits and candidate sets",
            "emit shared prediction schema",
        ),
    ),
    "llm_esr_qwen_lora": ReferenceMethodSpec(
        baseline_id="llm_esr_qwen_lora",
        title="LLM-ESR: Large Language Models Enhancement for Long-tailed Sequential Recommendation",
        local_reference=(
            "references/NH/NeurIPS-2024-llm-esr-large-language-models-enhancement-for-"
            "long-tailed-sequential-recommendation-Paper-Conference.pdf"
        ),
        public_url=(
            "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
            "2f0728449cb3150189d765fc87afc913-Abstract-Conference.html"
        ),
        official_code_url="https://github.com/Applied-Machine-Learning-Lab/LLM-ESR",
        official_code_status="official_code_identified",
        method_family="long_tail_sequential_recommendation",
        preserve_components=(
            "LLM semantic item/user signals",
            "semantic and collaborative dual-view modeling",
            "retrieval-augmented self-distillation",
            "long-tail user/item handling",
        ),
        fair_protocol_adaptations=(
            "derive semantic signals with the shared Qwen3-8B backbone when faithful",
            "use frozen TGL-Rec same-candidate evaluation",
            "report long-tail and ranking diagnostics",
        ),
    ),
    "controllable_rec_qwen_lora": ReferenceMethodSpec(
        baseline_id="controllable_rec_qwen_lora",
        title="Aligning Large Language Models for Controllable Recommendations",
        local_reference="references/NH/Aligning Large Language Models for Controllable Recommendations.pdf",
        public_url="https://arxiv.org/abs/2403.05063",
        official_code_url=None,
        official_code_status="no_official_code_identified",
        method_family="controllable_recommendation_alignment",
        preserve_components=(
            "recommendation-specific instruction tasks",
            "control labels or control constraints",
            "alignment objective for controllable recommendation",
            "format and candidate grounding constraints",
        ),
        fair_protocol_adaptations=(
            "use Qwen3-8B LoRA for instruction/control adaptation",
            "use shared candidates and evaluator",
            "preserve control-condition evaluation when available",
        ),
    ),
    "cllm4rec_qwen_lora": ReferenceMethodSpec(
        baseline_id="cllm4rec_qwen_lora",
        title="Collaborative Large Language Model for Recommender Systems",
        local_reference="references/NR/3589334.3645347.pdf",
        public_url="https://github.com/yaochenzhu/LLM4Rec",
        official_code_url="https://github.com/yaochenzhu/LLM4Rec",
        official_code_status="official_code_identified",
        method_family="collaborative_llm_recommendation",
        preserve_components=(
            "user and item ID tokens",
            "soft and hard prompting",
            "item prediction head",
            "mutual regularization between text and collaborative signals",
        ),
        fair_protocol_adaptations=(
            "attach collaborative ID/prompt structure to Qwen3-8B LoRA when feasible",
            "score the shared candidate set",
            "emit shared predictions with candidate adherence diagnostics",
        ),
    ),
    "rlmrec_qwen_lora": ReferenceMethodSpec(
        baseline_id="rlmrec_qwen_lora",
        title="Representation Learning with Large Language Models for Recommendation",
        local_reference="references/NR/3589334.3645458.pdf",
        public_url="https://github.com/HKUDS/RLMRec",
        official_code_url="https://github.com/HKUDS/RLMRec",
        official_code_status="official_code_identified",
        method_family="llm_representation_learning",
        preserve_components=(
            "LLM-generated user/item semantic profiles",
            "semantic representation learning",
            "collaborative representation learning",
            "cross-view alignment objective",
        ),
        fair_protocol_adaptations=(
            "use Qwen3-derived profiles or embeddings when faithful",
            "train/evaluate under shared splits and candidates",
            "preserve representation-alignment scoring logic",
        ),
    ),
    "transrec_qwen_lora": ReferenceMethodSpec(
        baseline_id="transrec_qwen_lora",
        title="Bridging Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation",
        local_reference="references/NR/3637528.3671884.pdf",
        public_url="https://arxiv.org/abs/2310.06491",
        official_code_url=None,
        official_code_status="no_official_code_identified",
        method_family="grounded_identifier_generation",
        preserve_components=(
            "multi-facet item identifiers",
            "transition from generated identifiers to in-corpus items",
            "grounded generation constraints",
        ),
        fair_protocol_adaptations=(
            "use Qwen3-8B LoRA for identifier generation if faithful",
            "ground outputs to the shared candidate set",
            "report validity and hallucination diagnostics",
        ),
    ),
    "review_pref_reasoning_qwen_lora": ReferenceMethodSpec(
        baseline_id="review_pref_reasoning_qwen_lora",
        title="Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation",
        local_reference="references/NR/3726302.3730055.pdf",
        public_url="https://arxiv.org/abs/2408.06276",
        official_code_url="https://github.com/jieyong99/EXP3RT",
        official_code_status="official_code_identified",
        method_family="review_preference_reasoning",
        preserve_components=(
            "review-to-preference extraction",
            "personalized preference reasoning",
            "LLM reranking based on extracted preferences",
        ),
        fair_protocol_adaptations=(
            "use Qwen3-8B LoRA for preference reasoning when faithful",
            "use shared candidates and evaluator",
            "preserve review-derived evidence when the dataset provides reviews",
        ),
    ),
}


def reference_method_names() -> list[str]:
    """Return selected reference method IDs in deterministic order."""

    return sorted(REFERENCE_METHOD_REGISTRY)


def get_reference_method(name: str) -> ReferenceMethodSpec:
    """Return one selected reference method spec."""

    key = str(name)
    if key not in REFERENCE_METHOD_REGISTRY:
        supported = ", ".join(reference_method_names())
        raise ValueError(f"Unsupported reference method: {name}. Supported methods: {supported}")
    return REFERENCE_METHOD_REGISTRY[key]


def require_implemented_reference_method(name: str) -> ReferenceMethodSpec:
    """Guard against accidentally training or reporting unimplemented reference scaffolds."""

    spec = get_reference_method(name)
    if spec.official_code_status != "official_code_identified":
        raise ReferenceBaselineNotImplementedError(
            f"{name} has no verified official code path and cannot be a main baseline."
        )
    if spec.implementation_status != "implemented" or not spec.reportable_baseline:
        raise ReferenceBaselineNotImplementedError(
            f"{name} is selected for faithful adaptation but is not implemented/reportable yet."
        )
    return spec
