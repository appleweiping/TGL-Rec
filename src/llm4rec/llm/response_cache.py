"""On-disk response cache for deterministic LLM diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.llm.base import LLMRequest, LLMResponse


class ResponseCache:
    """JSON response cache keyed by model, prompt, decoding params, and candidates."""

    def __init__(self, cache_dir: str | Path = "outputs/cache/llm", *, enabled: bool = True):
        self.cache_dir = ensure_dir(resolve_path(cache_dir))
        self.enabled = bool(enabled)

    def key_for(self, request: LLMRequest) -> str:
        """Build a stable cache key for a request without storing secrets."""

        payload = {
            "candidate_item_ids": [str(item) for item in request.candidate_item_ids],
            "dataset_run_id": request.metadata.get("dataset_run_id"),
            "decoding_params": request.decoding_params,
            "model": request.model,
            "prompt_hash": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "prompt_version": request.prompt_version,
            "provider": request.provider,
        }
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, request: LLMRequest) -> LLMResponse | None:
        """Return a cached response when available."""

        if not self.enabled:
            return None
        path = self.path_for(request)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(
            raw_output=str(data["raw_output"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cache_hit=True,
            metadata=dict(data.get("metadata", {})),
        )

    def set(self, request: LLMRequest, response: LLMResponse) -> Path:
        """Persist a response in the cache."""

        path = self.path_for(request)
        if not self.enabled:
            return path
        write_json(
            path,
            {
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
                "metadata": response.metadata,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "provider": response.provider,
                "raw_output": response.raw_output,
                "total_tokens": response.total_tokens,
            },
        )
        return path

    def path_for(self, request: LLMRequest) -> Path:
        """Return cache path for a request."""

        return self.cache_dir / f"{self.key_for(request)}.json"

