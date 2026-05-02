"""Vectorized TemporalGraphEncoder scorer for compact shared-pool candidates."""

from __future__ import annotations

from typing import Any

from llm4rec.scoring.candidate_batch import CandidateBatch


class TemporalGraphSharedPoolScorer:
    """Score shared-pool candidates with batched temporal user and item embeddings."""

    name = "temporal_graph_encoder_shared_pool"

    def __init__(self, *, encoder: Any, torch_module: Any) -> None:
        self.encoder = encoder
        self.torch = torch_module
        self.metadata = {
            "hidden_dim": int(getattr(encoder, "hidden_dim", 0)),
            "scorer": self.name,
            "vectorized": True,
        }

    @classmethod
    def from_context(cls, context: Any) -> "TemporalGraphSharedPoolScorer":
        """Build from the paper-matrix method context."""

        return cls(
            encoder=context.state["temporal_encoder"],
            torch_module=context.state["torch"],
        )

    def score_batch(self, batch: CandidateBatch) -> Any:
        torch = self.torch
        user_indices = [self.encoder.user_to_idx.get(str(user_id), 0) for user_id in batch.user_ids]
        candidate_indices = [
            [self.encoder.item_to_idx.get(str(item), 0) for item in row]
            for row in batch.candidate_item_ids
        ]
        timestamps = [0.0 if value is None else float(value) for value in batch.prediction_timestamps]
        with torch.no_grad():
            user_tensor = torch.tensor(user_indices, dtype=torch.long)
            item_tensor = torch.tensor(candidate_indices, dtype=torch.long)
            time_tensor = torch.tensor(timestamps, dtype=torch.float32).view(-1, 1)
            time_scale = torch.log1p(time_tensor.clamp(min=0.0))
            user_vec = self.encoder.user_memory(user_tensor) + self.encoder.time_projection(time_scale)
            item_vec = self.encoder.item_memory(item_tensor)
            return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1).detach().cpu().numpy()
