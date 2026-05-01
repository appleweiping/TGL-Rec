"""Cost and token preflight estimates for API micro diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostPreflight:
    """Audit-safe cost preflight summary saved before API execution."""

    number_of_cases: int
    prompt_variants: list[str]
    estimated_api_calls: int
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    max_api_calls: int
    cache_enabled: bool
    cache_policy: str
    model_name: str
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable dictionary."""

        return {
            "cache_enabled": self.cache_enabled,
            "cache_policy": self.cache_policy,
            "estimated_api_calls": self.estimated_api_calls,
            "estimated_completion_tokens": self.estimated_completion_tokens,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "max_api_calls": self.max_api_calls,
            "model_name": self.model_name,
            "number_of_cases": self.number_of_cases,
            "prompt_variants": self.prompt_variants,
            "run_dir": self.run_dir,
        }


def estimate_token_count(text: str) -> int:
    """Estimate token count conservatively without provider-specific tokenizers."""

    value = str(text)
    if not value:
        return 0
    char_estimate = (len(value) + 3) // 4
    whitespace_estimate = len(value.split())
    return max(1, max(char_estimate, whitespace_estimate))


def build_cost_preflight(
    *,
    prompts: list[str],
    number_of_cases: int,
    prompt_variants: list[str],
    max_tokens: int,
    max_api_calls: int,
    cache_enabled: bool,
    cache_policy: str,
    model_name: str,
    run_dir: str,
) -> CostPreflight:
    """Build a preflight estimate for planned prompt requests."""

    estimated_api_calls = len(prompts)
    return CostPreflight(
        number_of_cases=int(number_of_cases),
        prompt_variants=[str(name) for name in prompt_variants],
        estimated_api_calls=int(estimated_api_calls),
        estimated_prompt_tokens=sum(estimate_token_count(prompt) for prompt in prompts),
        estimated_completion_tokens=int(max_tokens) * int(estimated_api_calls),
        max_api_calls=int(max_api_calls),
        cache_enabled=bool(cache_enabled),
        cache_policy=str(cache_policy),
        model_name=str(model_name),
        run_dir=str(run_dir),
    )


def assert_within_call_cap(preflight: CostPreflight) -> None:
    """Enforce the hard API call cap before any possible API request."""

    if preflight.max_api_calls <= 0:
        raise ValueError("max_api_calls must be a positive integer.")
    if preflight.estimated_api_calls > preflight.max_api_calls:
        raise ValueError(
            "Estimated API calls exceed max_api_calls: "
            f"{preflight.estimated_api_calls} > {preflight.max_api_calls}"
        )
