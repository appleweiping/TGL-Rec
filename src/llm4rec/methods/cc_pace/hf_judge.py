"""Concrete Qwen3-8B forced-choice evidence model (server / GPU only).

Isolated from ``judge.py`` so the rest of CC-PACE stays importable and testable
on CPU without ``torch``/``transformers``. Loaded lazily.
"""

from __future__ import annotations


class HFForcedChoiceModel:
    """Reads per-label log-probabilities from a frozen Qwen3-8B.

    For each label id (e.g. ``[017]``), we score the model's probability of
    emitting that label string immediately after the prompt's "Best label:" cue,
    length-normalized over the label's tokens.

    Cost model: the long panel prompt is prefilled ONCE per call (KV cache);
    each label suffix is then a ~5-token batch-1 forward reusing the cached
    prefix. One panel rendering = one ~10k-token prefill + 101 tiny forwards,
    instead of 101 full forwards.
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
        return score_label_logprobs(
            self.model,
            self.tok,
            prompt,
            label_ids,
            device=self.device,
            max_ctx=self.max_ctx,
        )


def score_label_logprobs(
    model,
    tok,
    prompt: str,
    label_ids: list[str],
    *,
    device: str = "cuda",
    max_ctx: int = 32768,
) -> list[float]:
    """Length-normalized label log-probs after the prompt, via one shared prefill.

    Module-level (model/tokenizer injected) so the math is testable on CPU with a
    tiny in-memory model against a naive per-label full-forward reference.
    """
    import torch

    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).input_ids.to(
        device
    )

    label_tok = [
        tok(label, return_tensors="pt", add_special_tokens=False).input_ids[0]
        for label in label_ids
    ]

    with torch.no_grad():
        # 1) single prefill of the shared prompt -> KV cache + last-position logits
        prefill = model(p_ids, use_cache=True)
        past = prefill.past_key_values
        p_len = p_ids.shape[1]
        first_logp_row = torch.log_softmax(prefill.logits[0, -1, :].float(), dim=-1)

        # 2) one short batch-1 forward per label, REUSING the prompt cache.
        # No cache expansion: DynamicCache.update materializes a full copy per
        # batch row (32 copies of a 10k-token 8B cache = OOM), so batching the
        # suffixes is a memory trap. Sequential 3-5-token forwards cost ~10ms
        # each; the single prefill dominates. After every forward the cache is
        # cropped back to the prompt length (update() appends in place).
        out: list[float] = [0.0] * len(label_ids)
        for idx, t in enumerate(label_tok):
            t_dev = t.unsqueeze(0).to(device)
            t_len = t.shape[0]
            attn = torch.ones((1, p_len + t_len), dtype=torch.long, device=device)
            logits = model(
                t_dev, attention_mask=attn, past_key_values=past, use_cache=True
            ).logits  # [1, t_len, V]
            _crop_cache(past, p_len)
            # token 0 is predicted from the prompt's last position (prefill logits);
            # token j>0 is predicted from suffix position j-1.
            lp = first_logp_row[t_dev[0, 0]]
            if t_len > 1:
                logp = torch.log_softmax(logits[0, : t_len - 1, :].float(), dim=-1)
                lp = lp + logp[torch.arange(t_len - 1), t_dev[0, 1:]].sum()
            out[idx] = float(lp.item()) / t_len  # length-normalized
        return out


def _crop_cache(past, length: int) -> None:
    """Crop an appended-to KV cache back to ``length`` tokens (no-op if shorter)."""
    seq = past.get_seq_length() if hasattr(past, "get_seq_length") else None
    if seq is not None and seq > length and hasattr(past, "crop"):
        past.crop(length)
