"""Local Hugging Face causal LM provider for LoRA reranking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.llm.base import LLMRequest, LLMResponse


@dataclass(frozen=True)
class HFLocalProviderConfig:
    """Local HF model/adapter configuration."""

    base_model_path: str
    adapter_path: str | None = None
    tokenizer_path: str | None = None
    trust_remote_code: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.0


class HFLocalProvider:
    """Lazy-loading local HF provider. Never downloads model weights."""

    provider_name = "hf_local"

    def __init__(self, config: HFLocalProviderConfig, *, dry_run: bool = False) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dry_run = bool(dry_run)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one local response, or deterministic dry-run JSON."""

        if self.dry_run:
            return LLMResponse(
                raw_output=json.dumps({"ranked_item_ids": request.candidate_item_ids[:10]}),
                provider=self.provider_name,
                model=str(self.config.base_model_path),
                metadata={"dry_run": True},
            )
        self._load()
        assert self.model is not None and self.tokenizer is not None
        started = time.perf_counter()
        inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=self.config.temperature > 0,
            max_new_tokens=self.config.max_new_tokens,
            temperature=max(self.config.temperature, 1e-5),
        )
        decoded = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return LLMResponse(
            raw_output=decoded,
            provider=self.provider_name,
            model=str(self.config.base_model_path),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"adapter_path": self.config.adapter_path},
        )

    def _load(self) -> None:
        if self.model is not None:
            return
        base = Path(self.config.base_model_path).expanduser()
        if not base.exists():
            raise FileNotFoundError(f"Local base model path does not exist: {base}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path or self.config.base_model_path,
            local_files_only=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path, local_files_only=True)
