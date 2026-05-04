"""Adaptive concurrency and retry/backoff helpers for API batches."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


class RetryableAPIError(RuntimeError):
    """Retryable HTTP/API failure."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class RateLimitConfig:
    """Adaptive concurrency settings."""

    max_concurrency: int = 32
    adaptive_concurrency: bool = True
    min_concurrency: int = 4
    max_concurrency_hard_cap: int = 128
    backoff_initial_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    jitter: bool = True
    max_retries: int = 8
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)


class AdaptiveConcurrencyController:
    """Track a soft concurrency limit and reduce it on 429 responses."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self.current_limit = min(
            int(config.max_concurrency),
            int(config.max_concurrency_hard_cap),
        )
        self.current_limit = max(int(config.min_concurrency), self.current_limit)
        self.rate_limit_events = 0

    def record_status(self, status: int | None) -> None:
        if status == 429:
            self.rate_limit_events += 1
            if self.config.adaptive_concurrency:
                self.current_limit = max(self.config.min_concurrency, self.current_limit // 2)
        elif status is not None and 200 <= int(status) < 300 and self.config.adaptive_concurrency:
            self.current_limit = min(
                self.config.max_concurrency_hard_cap,
                self.config.max_concurrency,
                self.current_limit + 1,
            )

    def backoff_seconds(self, attempt: int) -> float:
        delay = min(
            float(self.config.backoff_max_seconds),
            float(self.config.backoff_initial_seconds) * (2 ** max(0, int(attempt) - 1)),
        )
        if self.config.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    def report(self) -> dict[str, int | bool]:
        return {
            "adaptive_concurrency": self.config.adaptive_concurrency,
            "current_limit": self.current_limit,
            "max_concurrency": self.config.max_concurrency,
            "min_concurrency": self.config.min_concurrency,
            "rate_limit_events": self.rate_limit_events,
        }


async def sleep_for_backoff(controller: AdaptiveConcurrencyController, attempt: int) -> None:
    await asyncio.sleep(controller.backoff_seconds(attempt))
