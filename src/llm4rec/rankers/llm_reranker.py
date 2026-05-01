"""LLM reranker wrapper for Phase 3A diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json
from llm4rec.llm.base import BaseLLMProvider, LLMRequest
from llm4rec.prompts.base import PromptExample
from llm4rec.prompts.builder import build_prompt
from llm4rec.prompts.parsers import try_parse_llm_response
from llm4rec.rankers.base import RankingExample, RankingResult


class LLMReranker:
    """Candidate reranker backed by an LLM provider.

    This class is infrastructure only. It does not implement OursMethod and should be used by
    Phase 3A diagnostic runners with fixed candidates and fixed prompt variants.
    """

    name = "llm_reranker"

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        item_records: dict[str, dict[str, Any]],
        prompt_variant: str,
        run_mode: str,
        model: str,
        decoding_params: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.item_records = item_records
        self.prompt_variant = str(prompt_variant)
        self.run_mode = str(run_mode)
        self.model = str(model)
        self.decoding_params = decoding_params or {}

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        """No-op: Phase 3A LLM diagnostics are not trained."""

    def rank(self, example: RankingExample) -> RankingResult:
        """Rank candidates with the configured LLM provider."""

        prompt_example = PromptExample(
            user_id=example.user_id,
            history=example.history,
            target_item=example.target_item,
            candidate_items=example.candidate_items,
            domain=example.domain,
            item_records=self.item_records,
            history_rows=list(example.metadata.get("history_rows", [])),
            transition_evidence=list(example.metadata.get("transition_evidence", [])),
            time_window_evidence=list(example.metadata.get("time_window_evidence", [])),
            contrastive_evidence=list(example.metadata.get("contrastive_evidence", [])),
            metadata=example.metadata,
        )
        prompt = build_prompt(prompt_example, prompt_variant=self.prompt_variant)
        request = LLMRequest(
            prompt=prompt.prompt,
            prompt_version=prompt.prompt_version,
            candidate_item_ids=prompt.candidate_item_ids,
            provider=getattr(self.provider, "provider_name", "unknown"),
            model=self.model,
            decoding_params=self.decoding_params,
            metadata={
                **prompt.metadata,
                **example.metadata,
                "run_mode": self.run_mode,
            },
        )
        response = self.provider.generate(request)
        parsed = try_parse_llm_response(response.raw_output, candidate_items=example.candidate_items)
        scores = [float(len(parsed.ranked_item_ids) - index) for index, _ in enumerate(parsed.ranked_item_ids)]
        return RankingResult(
            user_id=example.user_id,
            items=parsed.ranked_item_ids,
            scores=scores,
            raw_output=response.raw_output,
            metadata={
                "invalid_item_ids": parsed.invalid_item_ids,
                "parse_success": parsed.parse_success,
                "prompt_variant": self.prompt_variant,
                "prompt_version": prompt.prompt_version,
            },
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        """Persist lightweight reranker metadata."""

        write_json(
            Path(output_dir) / "llm_reranker.json",
            {
                "model": self.model,
                "non_reportable": self.run_mode != "diagnostic_api",
                "prompt_variant": self.prompt_variant,
                "run_mode": self.run_mode,
            },
        )

