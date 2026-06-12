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
    all label suffixes are then scored in a single batched short forward that
    reuses the cached prefix. One panel rendering therefore costs one ~10k-token
    prefill + one batched ~5-token forward, instead of 101 full forwards.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_ctx: int = 32768,
        suffix_batch: int = 32,
    ) -> None:
        import torch  # noqa: F401  (import-time check)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = __import__("torch")
        self.device = device
        self.max_ctx = max_ctx
        self.suffix_batch = int(suffix_batch)
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
            suffix_batch=self.suffix_batch,
        )


def score_label_logprobs(
    model,
    tok,
    prompt: str,
    label_ids: list[str],
    *,
    device: str = "cuda",
    max_ctx: int = 32768,
    suffix_batch: int = 32,
) -> list[float]:
    """Length-normalized label log-probs after the prompt, via one shared prefill.

    Module-level (model/tokenizer injected) so the math is testable on CPU with a
    tiny in-memory model against a naive per-label full-forward reference.
    """
    import torch

    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).input_ids.to(
        device
    )

    # tokenize all labels; right-pad to a common length for one batched forward
    label_tok = [
        tok(label, return_tensors="pt", add_special_tokens=False).input_ids[0]
        for label in label_ids
    ]
    max_len = max(t.shape[0] for t in label_tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    with torch.no_grad():
        # 1) single prefill of the shared prompt -> KV cache + last-position logits
        prefill = model(p_ids, use_cache=True)
        first_logp_row = torch.log_softmax(prefill.logits[0, -1, :].float(), dim=-1)

        out: list[float] = [0.0] * len(label_ids)
        for start in range(0, len(label_tok), suffix_batch):
            chunk = label_tok[start : start + suffix_batch]
            bsz = len(chunk)
            suffix = torch.full((bsz, max_len), pad_id, dtype=torch.long)
            attn_suffix = torch.zeros((bsz, max_len), dtype=torch.long)
            for b, t in enumerate(chunk):
                suffix[b, : t.shape[0]] = t
                attn_suffix[b, : t.shape[0]] = 1
            suffix = suffix.to(device)
            attn = torch.cat(
                [
                    torch.ones((bsz, p_ids.shape[1]), dtype=torch.long, device=device),
                    attn_suffix.to(device),
                ],
                dim=1,
            )
            # 2) batched short forward over the cached prefix. _expand_past builds a
            # NEW cache from zero-copy .expand views, so the prompt cache is never
            # mutated and per-chunk memory stays transient (use_cache=False verified
            # to honour past_key_values on transformers 5.7: maxdiff 1e-7 vs naive).
            batch_past = _expand_past(prefill.past_key_values, bsz)
            logits = model(
                suffix, attention_mask=attn, past_key_values=batch_past, use_cache=False
            ).logits  # [bsz, max_len, V]
            logp = torch.log_softmax(logits.float(), dim=-1)
            for b, t in enumerate(chunk):
                t_dev = t.to(device)
                t_len = t_dev.shape[0]
                # token 0 is predicted from the prompt's last position (prefill logits);
                # token j>0 is predicted from suffix position j-1.
                lp = first_logp_row[t_dev[0]]
                if t_len > 1:
                    rows = logp[b, : t_len - 1, :]
                    lp = lp + rows[torch.arange(t_len - 1), t_dev[1:]].sum()
                out[start + b] = float(lp.item()) / t_len  # length-normalized
        return out


def _expand_past(past, batch_size: int):
    """Build a NEW batch-``batch_size`` cache from zero-copy views of a batch-1 cache.

    Never mutates ``past`` (DynamicCache.batch_repeat_interleave is IN-PLACE and
    returns None on transformers 5.x, so it must not be used on the shared prompt
    cache). Supports the 5.x ``cache.layers`` API and the legacy tuple format.
    """
    if hasattr(past, "layers"):  # transformers 5.x Cache API
        from transformers.cache_utils import DynamicCache

        new = DynamicCache()
        for li, layer in enumerate(past.layers):
            new.update(
                layer.keys.expand(batch_size, -1, -1, -1),
                layer.values.expand(batch_size, -1, -1, -1),
                li,
            )
        return new
    # legacy tuple-of-tuples format
    expanded = tuple(
        tuple(t.expand(batch_size, -1, -1, -1) for t in layer) for layer in past
    )
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(expanded)
    except Exception:
        return expanded
