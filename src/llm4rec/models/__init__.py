"""Trainable model definitions."""

from llm4rec.models.sasrec import SASRecModel, TORCH_AVAILABLE, TorchUnavailableError

__all__ = ["SASRecModel", "TORCH_AVAILABLE", "TorchUnavailableError"]
