"""Hard safety guard for Phase 3B API micro diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4rec.llm.cost_estimator import CostPreflight


class APIGuardError(RuntimeError):
    """Raised when a real API request is not allowed."""


@dataclass(frozen=True)
class APIGuardConfig:
    """Non-secret API guard inputs."""

    run_mode: str
    allow_api_calls: bool
    provider: str
    api_key_env: str
    max_api_calls: int | None
    estimated_calls: int
    cache_policy: str | None
    run_dir: Path
    resume: bool = False


def build_api_guard_config(
    *,
    config: dict[str, Any],
    llm_config: dict[str, Any],
    preflight: CostPreflight,
    run_dir: str | Path,
) -> APIGuardConfig:
    """Build guard config from resolved diagnostic and LLM configuration."""

    experiment = dict(config.get("experiment", {}))
    cache = dict(config.get("cache", {}))
    controls = dict(config.get("api_safety", {}))
    return APIGuardConfig(
        run_mode=str(experiment.get("run_mode", llm_config.get("run_mode", ""))),
        allow_api_calls=bool(llm_config.get("allow_api_calls", False)),
        provider=str(llm_config.get("provider", "")),
        api_key_env=str(llm_config.get("api_key_env", "")),
        max_api_calls=controls.get("max_api_calls", preflight.max_api_calls),
        estimated_calls=int(preflight.estimated_api_calls),
        cache_policy=cache.get("policy"),
        run_dir=Path(run_dir),
        resume=bool(experiment.get("resume", False)),
    )


def validate_api_guard(guard: APIGuardConfig) -> None:
    """Validate all conditions required before a real API request can happen."""

    if guard.run_mode != "diagnostic_api":
        raise APIGuardError("API calls require run_mode=diagnostic_api.")
    if not guard.allow_api_calls:
        raise APIGuardError("API calls require allow_api_calls=true.")
    if guard.provider == "mock":
        raise APIGuardError("API micro diagnostics cannot use provider=mock in API mode.")
    if guard.provider != "openai_compatible":
        raise APIGuardError(f"Unsupported API provider for Phase 3B: {guard.provider}")
    if not guard.api_key_env or not os.environ.get(guard.api_key_env):
        raise APIGuardError(f"Missing API key environment variable: {guard.api_key_env}")
    if guard.max_api_calls is None:
        raise APIGuardError("max_api_calls must be set.")
    max_calls = int(guard.max_api_calls)
    if max_calls <= 0:
        raise APIGuardError("max_api_calls must be positive.")
    if guard.estimated_calls > max_calls:
        raise APIGuardError(
            f"estimated_calls exceeds max_api_calls: {guard.estimated_calls} > {max_calls}"
        )
    if guard.cache_policy not in {"read_write", "read_only", "disabled"}:
        raise APIGuardError("cache policy must be explicit: read_write, read_only, or disabled.")
    _validate_run_dir_state(guard.run_dir, resume=guard.resume)


def validate_dry_run_config(guard: APIGuardConfig) -> list[str]:
    """Return warnings from the same guard inputs without requiring API credentials."""

    warnings: list[str] = []
    if guard.run_mode != "diagnostic_api":
        warnings.append("real API would be blocked because run_mode is not diagnostic_api")
    if not guard.allow_api_calls:
        warnings.append("real API would be blocked because allow_api_calls is false")
    if guard.provider == "mock":
        warnings.append("real API would be blocked because provider is mock")
    if not guard.api_key_env or not os.environ.get(guard.api_key_env):
        warnings.append(f"real API would be blocked because {guard.api_key_env} is not set")
    if guard.max_api_calls is None:
        warnings.append("real API would be blocked because max_api_calls is missing")
    elif guard.estimated_calls > int(guard.max_api_calls):
        warnings.append("real API would be blocked because estimated calls exceed max_api_calls")
    if guard.cache_policy not in {"read_write", "read_only", "disabled"}:
        warnings.append("real API would be blocked because cache policy is not explicit")
    try:
        _validate_run_dir_state(guard.run_dir, resume=guard.resume)
    except APIGuardError as exc:
        warnings.append(str(exc))
    return warnings


def _validate_run_dir_state(run_dir: Path, *, resume: bool) -> None:
    if resume or not run_dir.exists():
        return
    incomplete_markers = [
        run_dir / "api_raw_outputs.jsonl",
        run_dir / "predictions.jsonl",
        run_dir / "diagnostics" / "api_failures.jsonl",
    ]
    has_partial_api_state = any(
        path.exists() and path.stat().st_size > 0 for path in incomplete_markers
    )
    completed = (run_dir / "api_micro_summary.json").is_file()
    if has_partial_api_state and not completed:
        raise APIGuardError(f"run_dir contains incomplete API state and resume=false: {run_dir}")
