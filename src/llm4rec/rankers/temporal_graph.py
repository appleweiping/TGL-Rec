"""Candidate ranker for TemporalGraphEncoder checkpoints."""

from __future__ import annotations

from pathlib import Path

from llm4rec.encoders.temporal_graph_encoder import (
    TORCH_AVAILABLE,
    TemporalGraphEncoder,
    TemporalGraphTorchUnavailableError,
)
from llm4rec.rankers.base import RankingExample, RankingResult


class TemporalGraphRanker:
    """Rank candidates using a trained lightweight temporal graph encoder."""

    name = "temporal_graph_encoder"

    def __init__(self, checkpoint_path: str | Path) -> None:
        if not TORCH_AVAILABLE:
            raise TemporalGraphTorchUnavailableError("PyTorch is required to load TemporalGraphEncoder.")
        self.encoder = TemporalGraphEncoder.load(checkpoint_path)

    def rank(self, example: RankingExample) -> RankingResult:
        timestamp = example.metadata.get("prediction_timestamp")
        scores = {
            str(item): self.encoder.score(example.user_id, str(item), timestamp)
            for item in example.candidate_items
        }
        ordered = sorted(example.candidate_items, key=lambda item: (-float(scores[str(item)]), str(item)))
        return RankingResult(
            user_id=example.user_id,
            items=[str(item) for item in ordered],
            scores=[float(scores[str(item)]) for item in ordered],
            raw_output=None,
            metadata={"dynamic_encoder": "temporal_graph_encoder", "reportable": False},
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
