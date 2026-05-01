"""Minimal PyTorch SASRec-style sequential recommender."""

from __future__ import annotations

from typing import Any


try:  # Torch is an optional dependency under the project `models` extra.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only base envs.
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None


class TorchUnavailableError(ImportError):
    """Raised when a PyTorch component is requested without PyTorch installed."""


def require_torch() -> Any:
    """Return torch or raise a clear optional-dependency error."""

    if torch is None:
        raise TorchUnavailableError("PyTorch is required for SASRec. Install the project with `.[models]`.")
    return torch


def causal_attention_mask(seq_len: int, *, device: Any | None = None) -> Any:
    """Return a bool causal mask where future positions are masked."""

    require_torch()
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


if TORCH_AVAILABLE:

    class SASRecModel(nn.Module):
        """Small SASRec-style model with causal self-attention."""

        def __init__(
            self,
            *,
            num_items: int,
            hidden_dim: int = 32,
            num_layers: int = 1,
            num_heads: int = 1,
            dropout: float = 0.1,
            max_seq_len: int = 50,
            padding_idx: int = 0,
        ) -> None:
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads")
            self.num_items = int(num_items)
            self.hidden_dim = int(hidden_dim)
            self.max_seq_len = int(max_seq_len)
            self.padding_idx = int(padding_idx)
            self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=padding_idx)
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.item_embedding.weight[self.padding_idx].fill_(0.0)

        def forward(self, item_sequences: Any) -> Any:
            """Encode left-padded item sequences.

            Args:
                item_sequences: LongTensor with shape [batch, max_seq_len], where 0 is padding.
            """

            if item_sequences.ndim != 2:
                raise ValueError("item_sequences must have shape [batch, seq_len]")
            batch_size, seq_len = item_sequences.shape
            if seq_len > self.max_seq_len:
                raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
            positions = torch.arange(seq_len, device=item_sequences.device).unsqueeze(0).expand(batch_size, -1)
            hidden = self.item_embedding(item_sequences) + self.position_embedding(positions)
            hidden = self.dropout(hidden)
            padding_mask = item_sequences.eq(self.padding_idx)
            encoded = self.encoder(
                hidden,
                mask=causal_attention_mask(seq_len, device=item_sequences.device),
                src_key_padding_mask=padding_mask,
            )
            encoded = self.layer_norm(encoded)
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            return encoded

        def final_state(self, item_sequences: Any) -> Any:
            """Return the last non-padding state for each sequence."""

            encoded = self.forward(item_sequences)
            lengths = item_sequences.ne(self.padding_idx).sum(dim=1).clamp(min=1)
            gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
            return encoded.gather(dim=1, index=gather_index).squeeze(1)

        def score_items(self, item_sequences: Any, item_indices: Any) -> Any:
            """Score candidate item indices for each sequence."""

            state = self.final_state(item_sequences)
            item_vectors = self.item_embedding(item_indices)
            return (state.unsqueeze(1) * item_vectors).sum(dim=-1)

else:

    class SASRecModel:  # type: ignore[no-redef]
        """Placeholder that raises a clear optional-dependency error."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_torch()
