"""Sequential trainer placeholders and smoke Markov training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json
from llm4rec.rankers.sequential import MarkovTransitionRanker


def train_markov_transition(
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train the deterministic Markov transition smoke baseline."""

    root = Path(processed_dir)
    ranker = MarkovTransitionRanker()
    ranker.fit(read_jsonl(root / "train.jsonl"), read_jsonl(root / "items.jsonl"))
    out = ensure_dir(output_dir)
    ranker.save_artifact(out)
    metadata = {"method": ranker.name, "reportable": False, "status": "trained"}
    write_json(out / "training_metadata.json", metadata)
    return metadata


class SequentialTrainerInterface:
    """Formal sequential model trainer interface placeholder."""

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def train(self) -> None:
        raise NotImplementedError(
            f"{self.model_name} trainer is an interface only in Phase 4; do not report results yet."
        )
