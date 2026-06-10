"""Concrete Qwen3-8B forced-choice evidence model (server / GPU only).

Isolated from ``judge.py`` so the rest of CC-PACE stays importable and testable
on CPU without ``torch``/``transformers``. Loaded lazily.
"""

from __future__ import annotations

import math
from typing import Any


class HFForcedChoiceModel:
    """Reads per-label log-probabilities from a frozen Qwen3-8B.

    For each label id (e.g. ``[017]``), we score the model's probability of
    emitting that label string immediately after the prompt's "Best label:" cue,
    length-normalized over the label's tokens. This is a single forward pass per
    label suffix sharing the cached prompt prefix, so cost ~ one long prompt +
    n short suffixes per panel.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_ctx: int = 32768,
    ) -> None:
        import torch  # noqa: F401  (import-time check)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = __import__("torch")
        self.device = device
        self.max_ctx = max_ctx
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.truncation_side = "left"  # never cut the trailing "Best label:" cue
        if getattr(self.tok, "add_bos_token", False):
            self.tok.add_bos_token = False
        torch_dtype = getattr(self._torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        ).eval()

    def label_logprobs(self, prompt: str, label_ids: list[str]) -> list[float]:
        torch = self._torch
        p_ids = self.tok(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_ctx
        ).input_ids
        with torch.no_grad():
            out: list[float] = []
            for label in label_ids:
                t_ids = self.tok(label, return_tensors="pt", add_special_tokens=False).input_ids
                inp = torch.cat([p_ids, t_ids], dim=1).to(self.device)
                logits = self.model(inp).logits
                t_len = t_ids.shape[1]
                logp = torch.log_softmax(logits[0, -t_len - 1 : -1, :].float(), dim=-1)
                tgt = inp[0, -t_len:]
                tok_lp = logp[torch.arange(t_len), tgt]
                out.append(float(tok_lp.mean().item()))  # length-normalized
        return out
