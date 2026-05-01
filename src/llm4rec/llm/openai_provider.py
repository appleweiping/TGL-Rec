"""OpenAI-compatible API provider interface for opt-in diagnostics."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.response_cache import ResponseCache
from llm4rec.llm.safety import ensure_api_allowed
from llm4rec.llm.structured_output import openai_response_format


class OpenAICompatibleProvider:
    """Minimal chat-completions provider guarded by explicit API opt-in."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        run_mode: str,
        allow_api_calls: bool = False,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        cache: ResponseCache | None = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key_env = str(api_key_env)
        self.run_mode = str(run_mode)
        self.allow_api_calls = bool(allow_api_calls)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.cache = cache

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call a compatible `/chat/completions` endpoint when explicitly allowed."""

        ensure_api_allowed(
            run_mode=str(request.metadata.get("run_mode", self.run_mode)),
            allow_api_calls=self.allow_api_calls,
        )
        if self.cache is not None:
            cached = self.cache.get(request)
            if cached is not None:
                return cached
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        payload = self._payload(request)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                http_request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                llm_response = self._to_response(parsed, latency_ms=(time.perf_counter() - start) * 1000.0)
                if self.cache is not None:
                    self.cache.set(request, llm_response)
                return llm_response
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"OpenAI-compatible request failed after retries: {last_error}") from last_error

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        params = dict(request.decoding_params)
        payload = {
            "max_tokens": int(params.get("max_tokens", 512)),
            "messages": [{"content": request.prompt, "role": "user"}],
            "model": self.model,
            "temperature": float(params.get("temperature", 0.0)),
            "top_p": float(params.get("top_p", 1.0)),
        }
        response_format = openai_response_format(
            enabled=bool(request.metadata.get("structured_output_enabled", False)),
            strict=bool(request.metadata.get("structured_output_strict", True)),
        )
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _to_response(self, parsed: dict[str, Any], *, latency_ms: float) -> LLMResponse:
        usage = dict(parsed.get("usage", {}))
        choices = parsed.get("choices", [])
        raw_output = ""
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                raw_output = str(message.get("content", ""))
        return LLMResponse(
            raw_output=raw_output,
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            latency_ms=latency_ms,
            metadata={
                "finish_reason": choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None,
                "raw_response": parsed,
                "response_id": parsed.get("id"),
            },
        )

