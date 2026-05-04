"""Async batch execution for cache-first API LLM requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from llm4rec.llm.base import LLMRequest, LLMResponse
from llm4rec.llm.rate_limit import (
    AdaptiveConcurrencyController,
    RateLimitConfig,
    RetryableAPIError,
    sleep_for_backoff,
)


@dataclass(frozen=True)
class BatchResult:
    """One batch request outcome."""

    index: int
    request: LLMRequest
    response: LLMResponse | None
    status: str
    attempts: int
    error: str = ""


async def run_async_batch(
    requests: list[LLMRequest],
    *,
    generate: Callable[[LLMRequest], LLMResponse],
    rate_limit: RateLimitConfig,
    error_budget: int,
) -> tuple[list[BatchResult], dict[str, Any]]:
    """Run requests with bounded adaptive concurrency and retry/backoff."""

    controller = AdaptiveConcurrencyController(rate_limit)
    queue: asyncio.Queue[tuple[int, LLMRequest]] = asyncio.Queue()
    for index, request in enumerate(requests):
        queue.put_nowait((index, request))
    results: list[BatchResult] = []
    failures = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal failures
        while not queue.empty():
            index, request = await queue.get()
            result = await _run_one(index, request, generate=generate, controller=controller)
            async with lock:
                results.append(result)
                if result.status != "succeeded":
                    failures += 1
                    if failures > int(error_budget):
                        while not queue.empty():
                            queue.get_nowait()
                            queue.task_done()
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(controller.current_limit)]
    await queue.join()
    for task in workers:
        task.cancel()
    results.sort(key=lambda result: result.index)
    return results, {
        **controller.report(),
        "error_budget": int(error_budget),
        "failures": failures,
        "requests": len(requests),
    }


async def _run_one(
    index: int,
    request: LLMRequest,
    *,
    generate: Callable[[LLMRequest], LLMResponse],
    controller: AdaptiveConcurrencyController,
) -> BatchResult:
    last_error = ""
    for attempt in range(1, controller.config.max_retries + 2):
        try:
            response = await asyncio.to_thread(generate, request)
            controller.record_status(int(response.metadata.get("http_status", 200) or 200))
            return BatchResult(index=index, request=request, response=response, status="succeeded", attempts=attempt)
        except RetryableAPIError as exc:
            controller.record_status(exc.status)
            last_error = str(exc)
            if attempt > controller.config.max_retries:
                break
            await sleep_for_backoff(controller, attempt)
        except Exception as exc:
            last_error = str(exc)
            break
    return BatchResult(
        index=index,
        request=request,
        response=None,
        status="failed",
        attempts=controller.config.max_retries + 1,
        error=last_error,
    )
