"""DeepSeek V4 Flash OpenAI-compatible provider."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from llm4rec.llm.api_cache import APICache
from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.rate_limit import RetryableAPIError
from llm4rec.llm.safety import ensure_api_allowed


@dataclass(frozen=True)
class DeepSeekProviderConfig:
    """DeepSeek provider settings."""

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stream: bool = False
    thinking: str = "disabled"
    timeout: float = 120.0
    max_retries: int = 8
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)


class DeepSeekV4FlashProvider:
    """Synchronous provider for DeepSeek's OpenAI-compatible chat API."""

    provider_name = "deepseek"

    def __init__(
        self,
        config: DeepSeekProviderConfig | None = None,
        *,
        allow_api_calls: bool = False,
        run_mode: str = "diagnostic_api",
        cache: APICache | None = None,
    ) -> None:
        self.config = config or DeepSeekProviderConfig()
        self.model = self.config.model
        self.allow_api_calls = bool(allow_api_calls)
        self.run_mode = str(run_mode)
        self.cache = cache

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one chat completion without persisting secrets."""

        ensure_api_allowed(
            run_mode=str(request.metadata.get("run_mode", self.run_mode)),
            allow_api_calls=self.allow_api_calls,
        )
        if self.cache is not None:
            cached = self.cache.get(request)
            if cached is not None:
                return cached
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.config.api_key_env}")
        payload = self.payload_for(request)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            http_request = urllib.request.Request(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(http_request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
                status = int(getattr(response, "status", 200))
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            message = _safe_error_body(exc)
            if status in self.config.retry_on_status:
                raise RetryableAPIError(message or f"DeepSeek retryable status {status}", status=status) from exc
            raise RuntimeError(f"DeepSeek API failed with status {status}: {message}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise RetryableAPIError(f"DeepSeek transport error: {exc}", status=None) from exc
        parsed = json.loads(body)
        output = self._to_response(parsed, latency_ms=(time.perf_counter() - started) * 1000.0)
        output.metadata["http_status"] = status
        if self.cache is not None:
            self.cache.set(request, output)
        return output

    def payload_for(self, request: LLMRequest) -> dict[str, Any]:
        params = {**request.decoding_params}
        return {
            "max_tokens": int(params.get("max_tokens", self.config.max_tokens)),
            "messages": [
                {
                    "content": "You are a strict JSON recommendation reranker. Output compact JSON only.",
                    "role": "system",
                },
                {"content": request.prompt, "role": "user"},
            ],
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "stream": bool(params.get("stream", self.config.stream)),
            "temperature": float(params.get("temperature", self.config.temperature)),
            "thinking": {"type": str(params.get("thinking", self.config.thinking))},
            "top_p": float(params.get("top_p", self.config.top_p)),
        }

    def safe_config(self) -> dict[str, Any]:
        return {
            "api_key_env": self.config.api_key_env,
            "base_url": self.config.base_url,
            "max_retries": self.config.max_retries,
            "max_tokens": self.config.max_tokens,
            "model": self.config.model,
            "retry_on_status": list(self.config.retry_on_status),
            "stream": self.config.stream,
            "temperature": self.config.temperature,
            "thinking": self.config.thinking,
            "timeout": self.config.timeout,
            "top_p": self.config.top_p,
        }

    def _to_response(self, parsed: dict[str, Any], *, latency_ms: float) -> LLMResponse:
        usage = dict(parsed.get("usage", {}))
        choices = parsed.get("choices", [])
        raw_output = ""
        finish_reason = None
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            finish_reason = choices[0].get("finish_reason")
            if isinstance(message, dict):
                raw_output = str(message.get("content", ""))
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return LLMResponse(
            raw_output=raw_output,
            provider=self.provider_name,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0),
            latency_ms=latency_ms,
            metadata={
                "finish_reason": finish_reason,
                "response_id": parsed.get("id"),
            },
        )


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")[:1000]
    except Exception:
        return ""
