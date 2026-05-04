"""Sanitized cache for OpenAI-compatible API responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.llm.base import LLMRequest, LLMResponse


class APICache:
    """JSON cache keyed by non-secret request content."""

    def __init__(self, cache_dir: str | Path, *, enabled: bool = True, resume: bool = True) -> None:
        self.cache_dir = ensure_dir(resolve_path(cache_dir))
        self.enabled = bool(enabled)
        self.resume = bool(resume)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def key_for(self, request: LLMRequest) -> str:
        payload = {
            "candidate_item_ids": [str(item) for item in request.candidate_item_ids],
            "decoding_params": request.decoding_params,
            "model": request.model,
            "prompt_hash": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "prompt_version": request.prompt_version,
            "provider": request.provider,
        }
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, request: LLMRequest) -> LLMResponse | None:
        if not self.enabled or not self.resume:
            self.misses += 1
            return None
        path = self.path_for(request)
        if not path.is_file():
            self.misses += 1
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        self.hits += 1
        return LLMResponse(
            raw_output=str(data.get("raw_output", "")),
            provider=str(data.get("provider", request.provider)),
            model=str(data.get("model", request.model)),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
            cache_hit=True,
            metadata=dict(data.get("metadata", {})),
        )

    def set(self, request: LLMRequest, response: LLMResponse) -> Path:
        path = self.path_for(request)
        if not self.enabled:
            return path
        write_json(
            path,
            {
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
                "metadata": _sanitize(response.metadata),
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "provider": response.provider,
                "raw_output": response.raw_output,
                "total_tokens": response.total_tokens,
            },
        )
        self.writes += 1
        return path

    def path_for(self, request: LLMRequest) -> Path:
        return self.cache_dir / f"{self.key_for(request)}.json"

    def report(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "cache_dir": str(self.cache_dir),
            "enabled": self.enabled,
            "hit_rate": self.hits / float(total or 1),
            "hits": self.hits,
            "misses": self.misses,
            "resume": self.resume,
            "writes": self.writes,
        }


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if str(key).lower() not in {"authorization", "api_key", "api-key", "x-api-key"}
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value
