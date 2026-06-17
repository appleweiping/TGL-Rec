"""Concrete vLLM duel model for PaRC (server/GPU implementation).

Implements the ``PairwiseDuelModel`` protocol (``pairwise_prompt.py``) on top of
a single, persistent vLLM ``LLM`` instance loading a frozen Qwen3-8B. The duel
logit for a rendered A-vs-B prompt is read from the next-token logprobs of the
answer label tokens:

    duel_logit(prompt) = logprob("A") - logprob("B")

i.e. the model's log-odds that the candidate in slot A is the more likely next
interaction than the one in slot B (exactly the protocol's contract: >0 favours
A, <0 favours B, 0 = tie). This is the per-duel signal that
``pairwise_prompt.symmetrized_duel`` then A/B-swaps to cancel position bias.

THROUGHPUT (PaRC's whole point): the schedule emits MANY duels per user, and
PaRC's speed advantage over CC-PACE's 32k-token listwise panels is that every
duel is a short (~300-600 tok) prompt that vLLM can BATCH. ``duel_logits_batch``
runs ONE ``llm.generate`` over all prompts; calling ``duel_logit`` once per pair
sequentially would forfeit that batching, so the ranker/driver must route
batches of pairs through ``duel_logits_batch``.

Decoding (verified pony pattern, vLLM 0.10.2 on env ``qwen_vllm``):
    SamplingParams(temperature=0.0, max_tokens=<small>, logprobs=20,
                   guided_decoding=GuidedDecodingParams(choice=["A","B"]))
``guided_decoding=choice`` constrains the FIRST emitted token to exactly "A" or
"B", so both labels are guaranteed to appear in the top-`logprobs` table of the
first generated position; we read both logprobs from there. If a label is
missing from the (truncated) logprob table we fall back to a large negative
floor so the difference is still finite and well-ordered.

IMPORT SAFETY: vLLM is NEVER imported at module load. All ``vllm`` imports are
deferred into ``__init__`` / methods, so this file imports fine on a CPU box with
no vllm/torch installed (the unit tests import it and exercise the offline
fallback). The offline CPU fallback exists ONLY for import-testing / plumbing; it
is deterministic and clearly NOT for reporting.
"""

from __future__ import annotations

import math
import re
from typing import Any

from llm4rec.methods.parc.config import DEFAULT_PARC_CONFIG, PaRCConfig

# Floor used when a label token is absent from the truncated top-logprob table.
_MISSING_LOGPROB = -20.0


class VLLMDuelModel:
    """vLLM-backed ``PairwiseDuelModel``: log-odds A-over-B from next-token logprobs.

    Load ONCE (heavy): a single ``LLM`` + tokenizer is created in ``__init__`` and
    reused for every duel. Construct with ``offline=True`` (or simply on a host
    without vllm — auto-detected) to get the import-safe CPU stub used by tests.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        config: PaRCConfig | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 1024,
        max_tokens: int = 4,
        logprobs: int = 20,
        seed: int = 0,
        use_guided_choice: bool = True,
        offline: bool = False,
    ) -> None:
        self.config = config or DEFAULT_PARC_CONFIG
        self.model_path = model or self.config.backbone_model
        self.max_tokens = int(max_tokens)
        self.logprobs = int(logprobs)
        self.seed = int(seed)
        self.use_guided_choice = bool(use_guided_choice)
        # Provenance of how each duel logit was extracted (for the results JSON).
        self.extraction = "vllm_guided_choice" if use_guided_choice else "vllm_raw_logprobs"

        self._llm = None
        self._tokenizer = None
        self._sampling_params = None
        self.offline = bool(offline) or not _vllm_available()
        if self.offline:
            self.extraction = "offline_cpu_fallback_NOT_FOR_REPORTING"
            return

        # ----- vLLM init (lazy import; mirrors pony run_ccrp_v3 init) -----
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        self._llm = LLM(
            model=self.model_path,
            tokenizer=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            dtype="float16",
            enforce_eager=False,
            trust_remote_code=False,
            seed=self.seed,
        )
        self._tokenizer = self._llm.get_tokenizer()

        sampling_kwargs: dict[str, Any] = dict(
            temperature=0.0,
            max_tokens=self.max_tokens,
            logprobs=self.logprobs,
            seed=self.seed,
        )
        if self.use_guided_choice:
            # Constrain the first token to exactly "A"/"B" so both labels are
            # present in the top-logprob table of position 0 (pony guided pattern).
            from vllm.sampling_params import GuidedDecodingParams  # noqa: PLC0415

            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(choice=["A", "B"])
        self._sampling_params = SamplingParams(**sampling_kwargs)

    # ------------------------------------------------------------------ #
    # PairwiseDuelModel protocol
    # ------------------------------------------------------------------ #
    def duel_logit(self, prompt: str) -> float:
        """Single-prompt duel logit = logprob("A") - logprob("B").

        Convenience for the protocol; throughput-critical callers MUST use
        ``duel_logits_batch`` so vLLM batches the whole schedule in one pass.
        """
        return self.duel_logits_batch([prompt])[0]

    def duel_logits_batch(self, prompts: list[str]) -> list[float]:
        """ONE ``llm.generate`` over all prompts -> one duel logit each.

        This single batched call is PaRC's throughput advantage: a user's whole
        duel schedule (and ideally many users, user-contiguous for prefix-cache
        reuse) goes through vLLM in one go. Returns ``logprob(A) - logprob(B)``
        per prompt, in input order.
        """
        if not prompts:
            return []
        if self.offline:
            return [_offline_duel_logit(p) for p in prompts]

        rendered = [self._format(p) for p in prompts]
        outputs = self._llm.generate(rendered, self._sampling_params)
        return [self._logit_from_output(o) for o in outputs]

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _format(self, prompt: str) -> str:
        """Apply the tokenizer chat template generically (pony pattern)."""
        tok = self._tokenizer
        if not getattr(tok, "chat_template", None):
            return prompt
        msg = [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def _logit_from_output(self, output: Any) -> float:
        """Extract logprob(A)-logprob(B) from the FIRST generated position.

        vLLM ``RequestOutput.outputs[0].logprobs`` is a list (one entry per
        generated token); each entry maps token-id -> ``Logprob`` (with
        ``.decoded_token`` and ``.logprob``). With guided choice the first token
        is "A" or "B" and both candidates appear in that position's top table.
        """
        try:
            gen = output.outputs[0]
        except (AttributeError, IndexError):
            return 0.0
        step_logprobs = getattr(gen, "logprobs", None)
        if step_logprobs:
            lp_a, lp_b = _label_logprobs(step_logprobs[0])
            return float(lp_a - lp_b)
        # No logprobs returned (shouldn't happen with logprobs>0): fall back to the
        # decoded text — "A" -> +1, "B" -> -1, else 0.
        text = (getattr(gen, "text", "") or "").strip().upper()
        if text.startswith("A"):
            return 1.0
        if text.startswith("B"):
            return -1.0
        return 0.0


def _label_logprobs(position_logprobs: dict) -> tuple[float, float]:
    """Read logprob("A"), logprob("B") from one position's top-logprob dict.

    Keys are token ids; values are vLLM ``Logprob`` objects exposing
    ``.decoded_token`` and ``.logprob``. We match the decoded token (stripped,
    upper-cased) against the answer labels. Missing label -> ``_MISSING_LOGPROB``.
    """
    lp_a = _MISSING_LOGPROB
    lp_b = _MISSING_LOGPROB
    for entry in position_logprobs.values():
        tokstr = getattr(entry, "decoded_token", None)
        if tokstr is None:
            continue
        norm = tokstr.strip().upper()
        lp = float(getattr(entry, "logprob", _MISSING_LOGPROB))
        if norm == "A":
            lp_a = max(lp_a, lp)
        elif norm == "B":
            lp_b = max(lp_b, lp)
    return lp_a, lp_b


# --------------------------------------------------------------------------- #
# Import-safety helpers + offline CPU fallback (NOT for reporting).
# --------------------------------------------------------------------------- #
def _vllm_available() -> bool:
    """True iff ``vllm`` is importable, WITHOUT importing it eagerly at module load."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("vllm") is not None


_A_SEG_RE = re.compile(r"A:\s*(.*?)\s*B:", re.DOTALL)
_B_SEG_RE = re.compile(r"B:\s*(.*?)\s*Given", re.DOTALL)


def _offline_duel_logit(prompt: str) -> float:
    """Deterministic CPU stub used only when vllm is absent (import-testing).

    Mirrors the ranker's ``_MockPairwiseDuelModel`` heuristic (history-overlap
    difference) so the wiring is exercisable on CPU. NEVER reported.
    """
    head, _, tail = prompt.partition("Two candidate items:")
    hist_toks = set(head.lower().replace("-", " ").split())
    a_match = _A_SEG_RE.search(tail)
    b_match = _B_SEG_RE.search(tail)
    a_seg = a_match.group(1) if a_match else ""
    b_seg = b_match.group(1) if b_match else ""
    a = len(set(a_seg.lower().split()) & hist_toks)
    b = len(set(b_seg.lower().split()) & hist_toks)
    return float(a - b)


__all__ = ["VLLMDuelModel"]
