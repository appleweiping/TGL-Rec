"""Vectorized MF/BPR scorer for compact shared-pool candidates."""

from __future__ import annotations

from typing import Any

from llm4rec.scoring.candidate_batch import CandidateBatch


class MFSharedPoolScorer:
    """Gather candidate embeddings and score [B, C] with batched dot products."""

    name = "mf_bpr_shared_pool"

    def __init__(
        self,
        *,
        user_emb: Any,
        item_emb: Any,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
        torch_module: Any,
    ) -> None:
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.user_to_idx = {str(key): int(value) for key, value in user_to_idx.items()}
        self.item_to_idx = {str(key): int(value) for key, value in item_to_idx.items()}
        self.torch = torch_module
        self.metadata = {
            "embedding_dim": int(item_emb.embedding_dim),
            "scorer": self.name,
            "vectorized": True,
        }

    @classmethod
    def from_context(cls, context: Any) -> "MFSharedPoolScorer":
        """Build from the paper-matrix method context."""

        return cls(
            user_emb=context.state["user_emb"],
            item_emb=context.state["item_emb"],
            user_to_idx=context.state["user_to_idx"],
            item_to_idx=context.state["item_to_idx"],
            torch_module=context.state["torch"],
        )

    def score_batch(self, batch: CandidateBatch) -> Any:
        torch = self.torch
        user_indices = [self.user_to_idx.get(str(user_id), 0) for user_id in batch.user_ids]
        candidate_indices = [
            [self.item_to_idx.get(str(item), 0) for item in row]
            for row in batch.candidate_item_ids
        ]
        with torch.no_grad():
            users = torch.tensor(user_indices, dtype=torch.long)
            items = torch.tensor(candidate_indices, dtype=torch.long)
            user_vec = self.user_emb(users)
            item_vec = self.item_emb(items)
            return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1).detach().cpu().numpy()
