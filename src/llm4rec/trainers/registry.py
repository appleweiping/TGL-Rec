"""Trainer registry for Phase 4 scripts."""

from __future__ import annotations

from typing import Any

from llm4rec.trainers.traditional import TraditionalBaselineTrainer


def build_trainer(config: dict[str, Any]):
    """Build a trainer from config."""

    trainer_type = str(config.get("training", {}).get("type", "traditional_mf"))
    if trainer_type in {"traditional_mf", "mf", "bpr_mf"}:
        return TraditionalBaselineTrainer(config=config)
    raise ValueError(f"Unknown trainer type: {trainer_type}")
