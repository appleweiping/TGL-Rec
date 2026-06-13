"""vLLM-backed Qwen forced-choice evidence model (server / GPU only).

This backend mirrors ``hf_judge.HFForcedChoiceModel`` while relying on vLLM's
prefix cache to avoid a full Transformers forward for every candidate label.
The module intentionally does not import vLLM at module import time so CPU tests
and developer machines without vLLM can still import ``llm4rec``.
"""

from __future__ import annotations

from typing import Any


class VLLMForcedChoiceModel:
    """Reads per-label log-probabilities through vLLM prompt logprobs.

    The scoring contract is the same as ``HFForcedChoiceModel``:

    - tokenize the long prompt with left truncation and no BOS;
    - tokenize labels with ``add_special_tokens=False``;
    - score label token 0 from the prompt-final next-token distribution;
    - score label token j>0 from the previous label-token position;
    - return the mean log-probability over label tokens.

    vLLM exposes those values as prompt-token logprobs when the request prompt is
    ``prompt_tokens + label_tokens``. Sending all labels in one batch gives vLLM
    identical prefixes to cache and reuse.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = "cuda",
        max_ctx: int = 32768,
        prompt_logprobs: int = 1,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_num_seqs: int | None = None,
        max_model_len: int | None = None,
        enable_prefix_caching: bool = True,
        trust_remote_code: bool = True,
        **llm_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - exercised only without vLLM
            raise ImportError(
                "VLLMForcedChoiceModel requires the optional 'vllm' package. "
                "Use --judge hf or install/activate a vLLM CUDA environment."
            ) from exc

        self.max_ctx = int(max_ctx)
        self.prompt_logprobs = int(prompt_logprobs)
        self._SamplingParams = SamplingParams

        init_kwargs: dict[str, Any] = {
            "model": model_path,
            "tokenizer": model_path,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            # HF scores a max_ctx-token prompt and then appends a short label.
            # Give vLLM a small suffix budget so the same token IDs are legal.
            "max_model_len": int(max_model_len or (self.max_ctx + 16)),
        }
        if device is not None:
            init_kwargs["device"] = device
        if max_num_seqs is not None:
            init_kwargs["max_num_seqs"] = int(max_num_seqs)
        if enable_prefix_caching:
            init_kwargs["enable_prefix_caching"] = True
        init_kwargs.update(llm_kwargs)

        self.llm = _build_llm(LLM, init_kwargs)
        self.tok = self.llm.get_tokenizer()
        self.tok.truncation_side = "left"
        if getattr(self.tok, "add_bos_token", False):
            self.tok.add_bos_token = False

    def label_logprobs(self, prompt: str, label_ids: list[str]) -> list[float]:
        if not label_ids:
            return []

        prompt_ids = _tokenize_prompt(self.tok, prompt, self.max_ctx)
        label_token_ids = [_tokenize_label(self.tok, label) for label in label_ids]
        if any(len(ids) == 0 for ids in label_token_ids):
            empty = [label_ids[i] for i, ids in enumerate(label_token_ids) if len(ids) == 0]
            raise ValueError(f"Empty label tokenization for labels: {empty}")

        token_prompts = [prompt_ids + ids for ids in label_token_ids]
        params = _sampling_params(self._SamplingParams, self.prompt_logprobs)
        outputs = _generate_with_token_ids(self.llm, token_prompts, params)

        if len(outputs) != len(label_token_ids):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(label_token_ids)} labels"
            )

        scores: list[float] = []
        start = len(prompt_ids)
        for out, label_ids_for_one in zip(outputs, label_token_ids):
            rows = getattr(out, "prompt_logprobs", None)
            if rows is None:
                raise RuntimeError("vLLM output did not include prompt_logprobs")
            if len(rows) < start + len(label_ids_for_one):
                raise RuntimeError(
                    "vLLM prompt_logprobs shorter than prompt+label token IDs: "
                    f"{len(rows)} < {start + len(label_ids_for_one)}"
                )
            total = 0.0
            for offset, token_id in enumerate(label_ids_for_one):
                row_index = start + offset
                total += _extract_logprob(rows[row_index], token_id, row_index)
            scores.append(total / len(label_ids_for_one))
        return scores


def _build_llm(llm_cls, init_kwargs: dict[str, Any]):
    """Construct vLLM while tolerating older constructor signatures."""
    try:
        return llm_cls(**init_kwargs)
    except TypeError as exc:
        msg = str(exc)
        retry = dict(init_kwargs)
        changed = False
        for key in ("enable_prefix_caching", "device"):
            if key in retry and key in msg:
                retry.pop(key)
                changed = True
        if not changed:
            raise
        return llm_cls(**retry)


def _sampling_params(sampling_params_cls, prompt_logprobs: int):
    kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "prompt_logprobs": prompt_logprobs,
        "logprobs": 0,
    }
    try:
        return sampling_params_cls(**kwargs)
    except TypeError:
        kwargs.pop("logprobs", None)
        return sampling_params_cls(**kwargs)


def _generate_with_token_ids(llm, token_prompts: list[list[int]], sampling_params):
    inputs = [{"prompt_token_ids": ids} for ids in token_prompts]
    try:
        return llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    except TypeError:
        return llm.generate(
            prompts=None,
            prompt_token_ids=token_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )


def _tokenize_prompt(tok, prompt: str, max_ctx: int) -> list[int]:
    encoded = tok(prompt, truncation=True, max_length=max_ctx)
    return _as_token_id_list(encoded)


def _tokenize_label(tok, label: str) -> list[int]:
    encoded = tok(label, add_special_tokens=False)
    return _as_token_id_list(encoded)


def _as_token_id_list(encoded) -> list[int]:
    ids = getattr(encoded, "input_ids", None)
    if ids is None and isinstance(encoded, dict):
        ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _extract_logprob(row, token_id: int, row_index: int) -> float:
    if row is None:
        raise RuntimeError(f"Missing vLLM prompt_logprobs row at token position {row_index}")

    if isinstance(row, dict):
        val = row.get(token_id)
        if val is None:
            val = row.get(str(token_id))
        if val is not None:
            return _coerce_logprob(val)
        for key, candidate in row.items():
            try:
                if int(key) == int(token_id):
                    return _coerce_logprob(candidate)
            except (TypeError, ValueError):
                pass
            candidate_token_id = getattr(candidate, "token_id", None)
            if candidate_token_id is not None and int(candidate_token_id) == int(token_id):
                return _coerce_logprob(candidate)

    if isinstance(row, (list, tuple)):
        for candidate in row:
            candidate_token_id = getattr(candidate, "token_id", None)
            if candidate_token_id is None and isinstance(candidate, dict):
                candidate_token_id = candidate.get("token_id")
            if candidate_token_id is not None and int(candidate_token_id) == int(token_id):
                return _coerce_logprob(candidate)

    raise RuntimeError(
        "vLLM prompt_logprobs did not include the scored label token "
        f"{token_id} at token position {row_index}. Increase prompt_logprobs or "
        "check the installed vLLM prompt_logprobs semantics."
    )


def _coerce_logprob(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    logprob = getattr(value, "logprob", None)
    if logprob is not None:
        return float(logprob)
    return float(value)
