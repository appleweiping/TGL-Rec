"""Minimal registries for Phase 1 experiment components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """String-key registry for pluggable components."""

    def __init__(self) -> None:
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._items:
            raise ValueError(f"Duplicate registry key: {name}")
        self._items[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown registry key {name!r}; available: {available}") from exc
