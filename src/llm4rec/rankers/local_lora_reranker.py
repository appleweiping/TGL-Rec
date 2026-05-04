"""Local LoRA reranker with strict JSON parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm4rec.llm.base import BaseLLMProvider, LLMRequest
from llm4rec.llm.json_parser import LLMJSONParseError, parse_rerank_json


@dataclass(frozen=True)
class LocalLoRARerankExample:
    """One local LoRA reranking input."""

    user_id: str
    history: list[str]
    target_item: str
    candidate_items: list[str]
    method: str = "local_8b_lora_history_only"
    metadata: dict[str, Any] | None = None


class LocalLoRAReranker:
    """Rerank top-m candidates with a local adapter-backed provider."""

    def __init__(self, *, provider: BaseLLMProvider, model: str, variant: str) -> None:
        self.provider = provider
        self.model = str(model)
        self.variant = str(variant)

    def rank(self, example: LocalLoRARerankExample) -> dict[str, Any]:
        prompt = _prompt(example, self.variant)
        request = LLMRequest(
            prompt=prompt,
            prompt_version=self.variant,
            candidate_item_ids=example.candidate_items,
            provider=getattr(self.provider, "provider_name", "hf_local"),
            model=self.model,
        )
        response = self.provider.generate(request)
        try:
            parsed = parse_rerank_json(response.raw_output, candidate_item_ids=example.candidate_items)
            ranked = _complete(parsed.ranked_item_ids, example.candidate_items)
            parse_success = True
            invalid = parsed.invalid_item_ids
        except LLMJSONParseError:
            ranked = list(example.candidate_items)
            parse_success = False
            invalid = []
        return {
            "metadata": {"parse_success": parse_success, "variant": self.variant},
            "predicted_items": ranked[:10],
            "raw_output": response.raw_output,
            "scores": [1.0 / float(index + 1) for index, _ in enumerate(ranked[:10])],
            "invalid_item_ids": invalid,
        }


def _prompt(example: LocalLoRARerankExample, variant: str) -> str:
    evidence = ""
    if variant == "temporal_evidence_sft":
        evidence = "\nUse temporal transition, time-window, and contrastive evidence when ranking."
    return (
        "Return JSON only with ranked_item_ids.\n"
        f"History: {example.history}\nCandidates: {example.candidate_items}{evidence}\n"
        + json.dumps({"ranked_item_ids": example.candidate_items[:10]})
    )


def _complete(items: list[str], candidates: list[str]) -> list[str]:
    output = list(items)
    seen = set(output)
    for item in candidates:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output
