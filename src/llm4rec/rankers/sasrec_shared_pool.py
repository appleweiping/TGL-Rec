"""Vectorized SASRec scorer for compact shared-pool candidates."""

from __future__ import annotations

from typing import Any

from llm4rec.scoring.candidate_batch import CandidateBatch
from llm4rec.trainers.sasrec import left_pad


class SASRecSharedPoolScorer:
    """Score candidate matrices with one sequence forward pass per user batch."""

    name = "sasrec_shared_pool"

    def __init__(
        self,
        *,
        model: Any,
        item_to_idx: dict[str, int],
        max_seq_len: int,
        torch_module: Any,
    ) -> None:
        self.model = model
        self.item_to_idx = {str(key): int(value) for key, value in item_to_idx.items()}
        self.max_seq_len = int(max_seq_len)
        self.torch = torch_module
        self.metadata = {
            "max_seq_len": self.max_seq_len,
            "scorer": self.name,
            "vectorized": True,
        }

    @classmethod
    def from_context(cls, context: Any) -> "SASRecSharedPoolScorer":
        """Build from the paper-matrix method context."""

        return cls(
            model=context.state["model"],
            item_to_idx=context.state["item_to_idx"],
            max_seq_len=int(context.state["max_seq_len"]),
            torch_module=context.state["torch"],
        )

    def score_batch(self, batch: CandidateBatch) -> Any:
        torch = self.torch
        sequences = [
            left_pad([self.item_to_idx.get(str(item), 0) for item in history], self.max_seq_len)
            for history in batch.histories
        ]
        candidate_indices = [
            [self.item_to_idx.get(str(item), 0) for item in row]
            for row in batch.candidate_item_ids
        ]
        with torch.no_grad():
            seq_tensor = torch.tensor(sequences, dtype=torch.long)
            item_tensor = torch.tensor(candidate_indices, dtype=torch.long)
            return self.model.score_items(seq_tensor, item_tensor).detach().cpu().numpy()
