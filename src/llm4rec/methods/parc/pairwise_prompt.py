"""Pairwise duel prompt for PaRC.

Builds the SHORT (~300-600 tok) forced-comparison prompt

    "given history H, is item A or B the more likely next interaction?"

and the A/B-swap symmetrization that cancels position bias:

    s_ij = 0.5 * (logit(A=i, B=j) - logit(A=j, B=i))

where ``logit(A, B)`` is the model's log-odds that the item placed in slot A is
the more likely next interaction than the item in slot B. Presenting the SAME
pair in both orders and averaging removes any constant slot-A/slot-B preference
(the position-bias diagnostic is ``b_ij = 0.5*(logit(A=i) + logit(A=j))``, the
part that does NOT flip with the swap).

The actual LLM call is INJECTED via the ``PairwiseDuelModel`` protocol so the
importable core has no vLLM/GPU dependency and is unit-testable on CPU. The
server wires a concrete implementation that reuses pony's
``run_ccrp_v3_domain_seeded`` vLLM init + guided-decoding + chat-template +
parse pattern (a single short prompt per duel, batched user-contiguous for
prefix caching).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from llm4rec.methods.parc.config import PaRCConfig


@runtime_checkable
class PairwiseDuelModel(Protocol):
    """Return the log-odds that slot-A is the more likely next interaction than slot-B.

    ``prompt`` is the rendered duel prompt (see ``build_pairwise_prompt``).
    Implementations read the model's preference for the two answer labels
    ("A" vs "B") and return a real-valued logit:
        > 0  -> A favoured, < 0 -> B favoured, 0 -> tie.
    Concrete server impl derives this from the next-token logprobs of the two
    label tokens under guided decoding (pony pattern).
    """

    def duel_logit(self, prompt: str) -> float:
        ...


def truncate_title(title: str, max_chars: int) -> str:
    """Trim a title to ``max_chars`` (keep prompts short, bound token count)."""
    t = str(title).strip()
    if max_chars and len(t) > max_chars:
        return t[: max_chars - 1].rstrip() + "…"
    return t


def _history_block(history: list[str], cfg: PaRCConfig) -> str:
    hist = [truncate_title(h, cfg.title_max_chars) for h in history[-cfg.max_history_items:] if str(h).strip()]
    if not hist:
        return "(no prior history)"
    # most-recent-first, one per line (mirrors pony's hist_block framing)
    return "\n".join(f"- {h}" for h in reversed(hist))


def _item_line(item: dict[str, Any] | str, cfg: PaRCConfig) -> str:
    """Render one candidate as ``Title[ — desc]`` with truncation."""
    if isinstance(item, str):
        title, desc = item, ""
    else:
        title = item.get("title", item.get("keywords", ""))
        desc = item.get("description", item.get("desc", "")) if cfg.desc_max_chars else ""
    line = truncate_title(title, cfg.title_max_chars)
    if cfg.desc_max_chars and desc:
        line += " — " + truncate_title(desc, cfg.desc_max_chars)
    return line


def build_pairwise_prompt(
    history: list[str],
    item_a: dict[str, Any] | str,
    item_b: dict[str, Any] | str,
    cfg: PaRCConfig,
) -> str:
    """Build the short A-vs-B duel prompt.

    Neutral labels ("A"/"B") are used by default so the model cannot anchor on a
    label identity; the candidate identity is carried only by the item line.
    """
    hist = _history_block(history, cfg)
    a_line = _item_line(item_a, cfg)
    b_line = _item_line(item_b, cfg)
    label_a, label_b = ("A", "B") if cfg.neutral_labels else (a_line, b_line)
    return (
        "You are an expert recommendation system.\n\n"
        f"User purchase history (most recent first):\n{hist}\n\n"
        "Two candidate items:\n"
        f"A: {a_line}\n"
        f"B: {b_line}\n\n"
        "Given the purchase pattern, which candidate is the MORE likely next "
        "interaction? Consider category alignment, attribute match, and purchase "
        "trajectory.\n\n"
        f'Return ONLY JSON: {{"more_likely": "{label_a}"}}'
    )


def symmetrized_duel(
    history: list[str],
    item_i: dict[str, Any] | str,
    item_j: dict[str, Any] | str,
    model: PairwiseDuelModel,
    cfg: PaRCConfig,
) -> dict[str, float]:
    """Run the A/B swap and return the symmetrized comparison + bias diagnostic.

    Returns ``{"s_ij", "logit_ij", "logit_ji", "position_bias"}`` where:
      - ``logit_ij`` = duel_logit(A=i, B=j)
      - ``logit_ji`` = duel_logit(A=j, B=i)
      - ``s_ij`` = 0.5*(logit_ij - logit_ji)  (i-over-j preference, bias cancelled)
      - ``position_bias`` = 0.5*(logit_ij + logit_ji)  (constant slot-A pull; 0 if unbiased)

    Note: ``logit_ji`` is A-favours-j, so j-over-i in i/j terms; subtracting it
    doubles the genuine i>j signal and cancels the additive slot bias.
    """
    logit_ij = float(model.duel_logit(build_pairwise_prompt(history, item_i, item_j, cfg)))
    if not cfg.symmetrize:
        return {"s_ij": logit_ij, "logit_ij": logit_ij, "logit_ji": 0.0, "position_bias": 0.0}
    logit_ji = float(model.duel_logit(build_pairwise_prompt(history, item_j, item_i, cfg)))
    s_ij = 0.5 * (logit_ij - logit_ji)
    position_bias = 0.5 * (logit_ij + logit_ji)
    return {
        "s_ij": s_ij,
        "logit_ij": logit_ij,
        "logit_ji": logit_ji,
        "position_bias": position_bias,
    }
