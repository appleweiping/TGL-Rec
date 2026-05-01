"""Lightweight trainable temporal graph encoder option.

This module implements a small dynamic graph encoder inspired by event-memory recommenders. It is
not a full TGN implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.encoders.base import BaseDynamicGraphEncoder


try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency path.
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None


class TemporalGraphTorchUnavailableError(ImportError):
    """Raised when the trainable temporal graph encoder is requested without torch."""


def require_torch() -> Any:
    if torch is None:
        raise TemporalGraphTorchUnavailableError(
            "PyTorch is required for TemporalGraphEncoder. Install the project with `.[models]`."
        )
    return torch


if TORCH_AVAILABLE:

    class TemporalGraphEncoder(nn.Module, BaseDynamicGraphEncoder):
        """Small trainable dynamic graph encoder with event-time memory updates."""

        reportable = False

        def __init__(
            self,
            *,
            num_users: int,
            num_items: int,
            hidden_dim: int = 16,
            user_to_idx: dict[str, int] | None = None,
            item_to_idx: dict[str, int] | None = None,
        ) -> None:
            super().__init__()
            self.num_users = int(num_users)
            self.num_items = int(num_items)
            self.hidden_dim = int(hidden_dim)
            self.user_to_idx = dict(user_to_idx or {})
            self.item_to_idx = dict(item_to_idx or {})
            self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
            self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
            self.user_memory = nn.Embedding(self.num_users + 1, self.hidden_dim, padding_idx=0)
            self.item_memory = nn.Embedding(self.num_items + 1, self.hidden_dim, padding_idx=0)
            self.time_projection = nn.Linear(1, self.hidden_dim)
            self.user_update = nn.GRUCell(self.hidden_dim * 2 + 1, self.hidden_dim)
            self.item_update = nn.GRUCell(self.hidden_dim * 2 + 1, self.hidden_dim)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.normal_(self.user_memory.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.item_memory.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.user_memory.weight[0].fill_(0.0)
                self.item_memory.weight[0].fill_(0.0)

        def fit(
            self,
            events: list[dict[str, Any]],
            item_features: dict[str, Any] | None = None,
            user_features: dict[str, Any] | None = None,
        ) -> None:
            del item_features, user_features
            for event in sorted(events, key=_event_sort_key):
                self.update(event)

        def encode_user(self, user_id: str, timestamp: int | float | None) -> list[float]:
            del timestamp
            idx = self.user_to_idx.get(str(user_id), 0)
            return self.user_memory.weight[idx].detach().cpu().tolist()

        def encode_item(self, item_id: str, timestamp: int | float | None) -> list[float]:
            del timestamp
            idx = self.item_to_idx.get(str(item_id), 0)
            return self.item_memory.weight[idx].detach().cpu().tolist()

        def score(self, user_id: str, item_id: str, timestamp: int | float | None = None) -> float:
            user_idx = torch.tensor([self.user_to_idx.get(str(user_id), 0)], dtype=torch.long, device=self.user_memory.weight.device)
            item_idx = torch.tensor([self.item_to_idx.get(str(item_id), 0)], dtype=torch.long, device=self.item_memory.weight.device)
            time_value = torch.tensor([[0.0 if timestamp is None else float(timestamp)]], dtype=torch.float32, device=self.user_memory.weight.device)
            time_scale = torch.log1p(time_value.clamp(min=0.0))
            user_vec = self.user_memory(user_idx) + self.time_projection(time_scale)
            item_vec = self.item_memory(item_idx)
            return float((user_vec * item_vec).sum(dim=-1).detach().cpu().item())

        def score_tensor(self, user_indices: Any, item_indices: Any, timestamps: Any) -> Any:
            time_scale = torch.log1p(timestamps.float().view(-1, 1).clamp(min=0.0))
            user_vec = self.user_memory(user_indices) + self.time_projection(time_scale)
            item_vec = self.item_memory(item_indices)
            return (user_vec * item_vec).sum(dim=-1)

        def update(self, event: dict[str, Any]) -> None:
            user_idx = self.user_to_idx.get(str(event.get("user_id", "")), 0)
            item_idx = self.item_to_idx.get(str(event.get("item_id", "")), 0)
            if user_idx == 0 or item_idx == 0:
                return
            timestamp = 0.0 if event.get("timestamp") is None else float(event.get("timestamp"))
            time_value = torch.tensor([[timestamp]], dtype=torch.float32, device=self.user_memory.weight.device)
            time_scale = torch.log1p(time_value.clamp(min=0.0))
            time_vec = time_scale.expand(1, 1)
            with torch.no_grad():
                user_vec = self.user_memory.weight[user_idx].view(1, -1)
                item_vec = self.item_memory.weight[item_idx].view(1, -1)
                update_input = torch.cat([user_vec, item_vec, time_vec], dim=-1)
                self.user_memory.weight[user_idx] = self.user_update(update_input, user_vec).squeeze(0)
                self.item_memory.weight[item_idx] = self.item_update(update_input, item_vec).squeeze(0)

        def save(self, path: str | Path) -> None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "hidden_dim": self.hidden_dim,
                    "item_to_idx": self.item_to_idx,
                    "model_state": self.state_dict(),
                    "num_items": self.num_items,
                    "num_users": self.num_users,
                    "reportable": self.reportable,
                    "user_to_idx": self.user_to_idx,
                },
                output,
            )

        @classmethod
        def load(cls, path: str | Path) -> "TemporalGraphEncoder":
            checkpoint = torch.load(path, map_location="cpu")
            model = cls(
                num_users=int(checkpoint["num_users"]),
                num_items=int(checkpoint["num_items"]),
                hidden_dim=int(checkpoint["hidden_dim"]),
                user_to_idx={str(key): int(value) for key, value in checkpoint["user_to_idx"].items()},
                item_to_idx={str(key): int(value) for key, value in checkpoint["item_to_idx"].items()},
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            return model

else:

    class TemporalGraphEncoder(BaseDynamicGraphEncoder):  # type: ignore[no-redef]
        """Placeholder that raises a clear optional-dependency error."""

        reportable = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_torch()

        def fit(self, events: list[dict[str, Any]], item_features: dict[str, Any] | None = None, user_features: dict[str, Any] | None = None) -> None:
            require_torch()

        def encode_user(self, user_id: str, timestamp: int | float | None) -> list[float]:
            require_torch()

        def encode_item(self, item_id: str, timestamp: int | float | None) -> list[float]:
            require_torch()

        def update(self, event: dict[str, Any]) -> None:
            require_torch()

        def save(self, path: str | Path) -> None:
            require_torch()

        @classmethod
        def load(cls, path: str | Path) -> "TemporalGraphEncoder":
            require_torch()


def build_temporal_graph_mappings(
    train_interactions: list[dict[str, Any]],
    item_records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Build user and item mappings, reserving 0 for unknown/padding."""

    users = sorted({str(row["user_id"]) for row in train_interactions})
    items = sorted({str(row["item_id"]) for row in item_records})
    return (
        {user_id: index for index, user_id in enumerate(users, start=1)},
        {item_id: index for index, item_id in enumerate(items, start=1)},
    )


def _event_sort_key(event: dict[str, Any]) -> tuple[float, str, str]:
    return (
        float(event["timestamp"]) if event.get("timestamp") is not None else -1.0,
        str(event.get("user_id", "")),
        str(event.get("item_id", "")),
    )
