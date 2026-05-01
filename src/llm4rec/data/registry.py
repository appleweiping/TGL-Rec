"""Dataset registry for Phase 1."""

from __future__ import annotations

from llm4rec.experiments.registry import Registry

DATASETS = Registry()


def register_dataset(name: str, factory) -> None:
    DATASETS.register(name, factory)


def get_dataset(name: str):
    return DATASETS.get(name)
