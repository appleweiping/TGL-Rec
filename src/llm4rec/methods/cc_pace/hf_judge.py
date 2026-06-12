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
    label suffixes are scored in small batched forwards (bounded cache copies).
    If a chunk OOMs, the scorer halves the suffix batch and ultimately falls
    back to a batch-1 no-cache-output forward. The private cache is a read-only
    view wrapper, so suffix scoring does not materialize and retain a full copy
    of the prompt KV cache. One panel rendering = one
    ~10k-token prefill + ceil(101/B) short forwards, instead of 101 full forwards.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_ctx: int = 32768,
        suffix_batch: int = 4,
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
    suffix_batch: int = 4,
) -> list[float]:
    """Length-normalized label log-probs after the prompt, via one shared prefill.

    Label suffixes are scored in bounded batches of ``suffix_batch``. Each chunk
    materializes copies of the prompt KV cache, so the runtime treats the batch
    size as an upper bound: CUDA OOM halves the active batch size, clears any
    partially built cache, and eventually falls back to batch-1 scoring. The
    scorer passes a read-only private view of the shared prompt cache to the
    model, so even Transformers cache implementations that call ``update`` with
    ``use_cache=False`` cannot mutate the shared prefill cache or retain a
    prompt-sized copy across layers.

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
        batch = max(1, int(suffix_batch))
        start = 0
        while start < len(label_tok):
            cur_batch = min(batch, len(label_tok) - start)
            while True:
                chunk = label_tok[start : start + cur_batch]
                try:
                    vals = _score_chunk_batched(
                        model, chunk, past, p_len, first_logp_row, pad_id, device
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    _clear_cuda_after_oom(torch)
                    if cur_batch <= 1:
                        try:
                            vals = [
                                _score_one_sequential(
                                    model, chunk[0], past, p_len, first_logp_row, device
                                )
                            ]
                        except torch.cuda.OutOfMemoryError:
                            _clear_cuda_after_oom(torch)
                            raise
                        break
                    cur_batch = max(1, cur_batch // 2)
                    batch = cur_batch
            out[start : start + len(vals)] = vals
            start += len(vals)
        _clear_cuda_after_oom(torch)
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
    batch_past = None
    try:
        batch_past = _expand_past(past, bsz)  # materialized copy, freed after the chunk
        logits = model(
            suffix, attention_mask=attn, past_key_values=batch_past, use_cache=False
        ).logits  # [bsz, max_len, V]
    except torch.cuda.OutOfMemoryError:
        if batch_past is not None:
            del batch_past
        del suffix, attn
        _clear_cuda_after_oom(torch)
        raise
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
    del batch_past, suffix, attn, logits, logp
    return vals


def _score_one_sequential(model, t, past, p_len, first_logp_row, device):
    """Batch-1 fallback using a private cache so shared prompt cache stays clean."""
    import torch

    t_dev = t.unsqueeze(0).to(device)
    t_len = t.shape[0]
    attn = torch.ones((1, p_len + t_len), dtype=torch.long, device=device)
    single_past = None
    try:
        single_past = _expand_past(past, 1)
        logits = model(
            t_dev, attention_mask=attn, past_key_values=single_past, use_cache=False
        ).logits
    except torch.cuda.OutOfMemoryError:
        if single_past is not None:
            del single_past
        del t_dev, attn
        _clear_cuda_after_oom(torch)
        raise
    lp = first_logp_row[t_dev[0, 0]]
    if t_len > 1:
        logp = torch.log_softmax(logits[0, : t_len - 1, :].float(), dim=-1)
        lp = lp + logp[torch.arange(t_len - 1), t_dev[0, 1:]].sum()
    val = float(lp.item()) / t_len
    del single_past, t_dev, attn, logits
    return val


def _expand_past(past, batch_size: int):
    """Build a private batch-``batch_size`` read-only view of a batch-1 cache.

    Never mutates ``past`` (DynamicCache.batch_repeat_interleave is IN-PLACE and
    returns None on transformers 5.x). The returned wrapper's ``update`` method
    returns per-layer ``past + suffix`` tensors without storing them, avoiding
    prompt-cache copies that accumulate across all decoder layers.
    """
    import torch

    if hasattr(past, "layers"):  # transformers 5.x Cache API
        try:
            return _ReadOnlyExpandedCache(
                [
                    _ReadOnlyExpandedLayer(
                        layer.keys.expand(batch_size, -1, -1, -1),
                        layer.values.expand(batch_size, -1, -1, -1),
                    )
                    for layer in past.layers
                ]
            )
        except torch.cuda.OutOfMemoryError:
            _clear_cuda_after_oom(torch)
            raise
    expanded = tuple(
        tuple(t.expand(batch_size, -1, -1, -1) for t in layer) for layer in past
    )
    return _ReadOnlyExpandedCache(
        [_ReadOnlyExpandedLayer(layer[0], layer[1]) for layer in expanded]
    )


class _ReadOnlyExpandedLayer:
    """Layer cache view whose update returns K/V for attention without retaining it."""

    def __init__(self, keys, values) -> None:
        self.keys = keys
        self.values = values
        self.is_initialized = True

    def get_seq_length(self) -> int:
        return self.keys.shape[-2]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def update(self, key_states, value_states, *args, **kwargs):
        import torch

        return (
            torch.cat([self.keys, key_states], dim=-2),
            torch.cat([self.values, value_states], dim=-2),
        )


class _ReadOnlyExpandedCache:
    """Minimal Cache-compatible wrapper for Qwen/Llama attention scoring."""

    is_compileable = False

    def __init__(self, layers: list[_ReadOnlyExpandedLayer]) -> None:
        self.layers = layers

    def __len__(self) -> int:
        return len(self.layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        return self.layers[layer_idx].update(
            key_states,
            value_states,
            *args,
            **kwargs,
        )


def _clear_cuda_after_oom(torch_module) -> None:
    """Release CUDA cache fragments left by a failed cache expansion."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    import gc

    gc.collect()
    cuda.empty_cache()
    try:
        cuda.ipc_collect()
    except Exception:
        pass


def _crop_cache(past, length: int) -> None:
    """Crop an appended-to KV cache back to ``length`` tokens (no-op if shorter)."""
    seq = past.get_seq_length() if hasattr(past, "get_seq_length") else None
    if seq is not None and seq > length and hasattr(past, "crop"):
        past.crop(length)
