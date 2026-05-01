"""LLM provider contracts for diagnostic reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    """One LLM completion request with non-secret metadata."""

    prompt: str
    prompt_version: str
    candidate_item_ids: list[str]
    provider: str
    model: str
    decoding_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Raw LLM response plus audit-safe usage metadata."""

    raw_output: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(Protocol):
    """Minimal provider interface shared by mock and API backends."""

    provider_name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response."""

