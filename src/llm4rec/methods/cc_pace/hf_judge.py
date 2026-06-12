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
    label suffixes are scored in small batched forwards (bounded cache copies)
    with a sequential batch-1 fallback on OOM. One panel rendering = one
    ~10k-token prefill + ceil(101/B) short forwards, instead of 101 full forwards.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_ctx: int = 32768,
        suffix_batch: int = 8,
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
    suffix_batch: int = 8,
) -> list[float]:
    """Length-normalized label log-probs after the prompt, via one shared prefill.

    Label suffixes are scored in bounded batches of ``suffix_batch``: each chunk
    materializes ``suffix_batch`` copies of the prompt KV cache (~1.5GB each for
    a 10k-token 8B prompt — B=8 is ~12GB, safe on a 48GB card; B=32+ OOMs). On
    CUDA OOM the chunk falls back to sequential batch-1 scoring transparently.

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
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    with torch.no_grad():
        # 1) single prefill of the shared prompt -> KV cache + last-position logits
        prefill = model(p_ids, use_cache=True)
        past = prefill.past_key_values
        p_len = p_ids.shape[1]
        first_logp_row = torch.log_softmax(prefill.logits[0, -1, :].float(), dim=-1)

        out: list[float] = [0.0] * len(label_ids)
        for start in range(0, len(label_tok), max(1, suffix_batch)):
            chunk = label_tok[start : start + max(1, suffix_batch)]
            try:
                vals = _score_chunk_batched(
                    model, chunk, past, p_len, first_logp_row, pad_id, device
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                vals = [
                    _score_one_sequential(model, t, past, p_len, first_logp_row, device)
                    for t in chunk
                ]
            out[start : start + len(vals)] = vals
        return out


def _score_chunk_batched(model, chunk, past, p_len, first_logp_row, pad_id, device):
    """Score a chunk of label suffixes in one batched forward over an expanded cache."""
    import torch

    bsz = len(chunk)
    max_len = max(t.shape[0] for t in chunk)
    suffix = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn_suffix = torch.zeros((bsz, max_len), dtype=torch.long)
    for b, t in enumerate(chunk):
        suffix[b, : t.shape[0]] = t
        attn_suffix[b, : t.shape[0]] = 1
    suffix = suffix.to(device)
    attn = torch.cat(
        [torch.ones((bsz, p_len), dtype=torch.long, device=device), attn_suffix.to(device)],
        dim=1,
    )
    batch_past = _expand_past(past, bsz)  # materialized copy, freed after the chunk
    logits = model(
        suffix, attention_mask=attn, past_key_values=batch_past, use_cache=False
    ).logits  # [bsz, max_len, V]
    del batch_past
    logp = torch.log_softmax(logits.float(), dim=-1)
    vals = []
    for b, t in enumerate(chunk):
        t_dev = t.to(device)
        t_len = t_dev.shape[0]
        # token 0 is predicted from the prompt's last position (prefill logits);
        # token j>0 is predicted from suffix position j-1.
        lp = first_logp_row[t_dev[0]]
        if t_len > 1:
            rows = logp[b, : t_len - 1, :]
            lp = lp + rows[torch.arange(t_len - 1), t_dev[1:]].sum()
        vals.append(float(lp.item()) / t_len)  # length-normalized
    return vals


def _score_one_sequential(model, t, past, p_len, first_logp_row, device):
    """Batch-1 fallback reusing (and cropping) the shared prompt cache."""
    import torch

    t_dev = t.unsqueeze(0).to(device)
    t_len = t.shape[0]
    attn = torch.ones((1, p_len + t_len), dtype=torch.long, device=device)
    logits = model(t_dev, attention_mask=attn, past_key_values=past, use_cache=True).logits
    _crop_cache(past, p_len)
    lp = first_logp_row[t_dev[0, 0]]
    if t_len > 1:
        logp = torch.log_softmax(logits[0, : t_len - 1, :].float(), dim=-1)
        lp = lp + logp[torch.arange(t_len - 1), t_dev[0, 1:]].sum()
    return float(lp.item()) / t_len


def _expand_past(past, batch_size: int):
    """Build a NEW batch-``batch_size`` cache from a batch-1 cache (materialized).

    Never mutates ``past`` (DynamicCache.batch_repeat_interleave is IN-PLACE and
    returns None on transformers 5.x). Memory cost is batch_size x prompt-cache
    — callers must keep batch_size small (see score_label_logprobs).
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
    expanded = tuple(
        tuple(t.expand(batch_size, -1, -1, -1) for t in layer) for layer in past
    )
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(expanded)
    except Exception:
        return expanded


def _crop_cache(past, length: int) -> None:
    """Crop an appended-to KV cache back to ``length`` tokens (no-op if shorter)."""
    seq = past.get_seq_length() if hasattr(past, "get_seq_length") else None
    if seq is not None and seq > length and hasattr(past, "crop"):
        past.crop(length)
