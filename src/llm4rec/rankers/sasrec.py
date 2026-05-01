"""SASRec candidate ranker wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.models.sasrec import SASRecModel, TORCH_AVAILABLE, TorchUnavailableError
from llm4rec.rankers.base import RankingExample, RankingResult
from llm4rec.trainers.sasrec import left_pad


class SASRecRanker:
    """Rank candidates with a trained SASRec checkpoint."""

    name = "sasrec"

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        if not TORCH_AVAILABLE:
            raise TorchUnavailableError("PyTorch is required to load SASRec checkpoints.")
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=device)
        training = checkpoint.get("config", {}).get("training", {})
        self.item_to_idx = {str(key): int(value) for key, value in checkpoint["item_to_idx"].items()}
        self.idx_to_item = {int(key): str(value) for key, value in checkpoint["idx_to_item"].items()}
        self.max_seq_len = int(training.get("max_seq_len", 20))
        self.device = torch.device(device)
        self.model = SASRecModel(
            num_items=len(self.item_to_idx),
            hidden_dim=int(training.get("hidden_dim", 32)),
            num_layers=int(training.get("num_layers", 1)),
            num_heads=int(training.get("num_heads", 1)),
            dropout=float(training.get("dropout", 0.1)),
            max_seq_len=self.max_seq_len,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def rank(self, example: RankingExample) -> RankingResult:
        import torch

        history = [self.item_to_idx[item] for item in example.history if item in self.item_to_idx]
        candidates = [str(item) for item in example.candidate_items]
        candidate_indices = [[self.item_to_idx.get(item, 0) for item in candidates]]
        sequence = torch.tensor([left_pad(history, self.max_seq_len)], dtype=torch.long, device=self.device)
        items = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self.model.score_items(sequence, items).squeeze(0).detach().cpu().tolist()
        ordered = sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), pair[0]))
        return RankingResult(
            user_id=example.user_id,
            items=[item for item, _score in ordered],
            scores=[float(score) for _item, score in ordered],
            raw_output=None,
            metadata={"reportable": False, "sequential_baseline": "sasrec"},
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
