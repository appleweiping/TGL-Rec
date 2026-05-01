"""Safety gates for diagnostic LLM providers."""

from __future__ import annotations


class LLMProviderSafetyError(RuntimeError):
    """Raised when a provider is used outside its allowed run mode."""


def ensure_mock_allowed(run_mode: str) -> None:
    """Forbid mock providers in reportable runs."""

    if str(run_mode) not in {"smoke", "diagnostic_mock"}:
        raise LLMProviderSafetyError(
            "MockLLMProvider is allowed only for run_mode=smoke or diagnostic_mock."
        )


def ensure_api_allowed(*, run_mode: str, allow_api_calls: bool) -> None:
    """Require explicit opt-in before real API calls."""

    if str(run_mode) != "diagnostic_api" or not bool(allow_api_calls):
        raise LLMProviderSafetyError(
            "OpenAI-compatible API calls require run_mode=diagnostic_api and allow_api_calls=true."
        )

