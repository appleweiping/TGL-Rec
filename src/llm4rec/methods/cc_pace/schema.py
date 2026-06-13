"""Unified candidate-schema rendering for CC-PACE.

Each panel candidate is rendered as a compact, uniform block:

    [LABEL]
    category: ...
    brand: ...
    keywords: ...
    attrs: ...
    collaborative: <neighbour evidence>   # only if use_cf_tokens

Label ids are RANDOMIZED per panel (a fresh permutation each rendering), which
makes the judge's joint scoring function permutation-equivariant over the panel.
That permutation-equivariance is exactly the exchangeability condition the
conformal layer relies on, and it removes position / name-leakage artefacts.

The user block carries long-term PROFILE SLOTS (skin type, concerns, brand
loyalty, routine step, ingredient prefs, price band). These are mandatory for
beauty, where the SOTA baseline (promax) wins precisely because beauty
preference is profile-expressible (docs/method_v2_decision_CC-PACE.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from llm4rec.methods.cc_pace.config import CCPaceConfig


@dataclass(frozen=True)
class RenderedPanel:
    """A label-randomized rendering of one user's 101-candidate panel."""

    user_block: str
    candidate_blocks: list[str]          # in PRESENTATION (shuffled) order
    label_ids: list[str]                 # label shown for each presented block
    presentation_to_original: list[int]  # presented position -> original index
    original_to_label: dict[int, str]    # original candidate index -> its label id


def _fmt_item_fields(item: dict[str, Any], schema_fields: tuple[str, ...]) -> str:
    lines = []
    for fld in schema_fields:
        val = item.get(fld, "")
        if isinstance(val, (list, tuple)):
            val = "; ".join(str(v) for v in val)
        lines.append(f"{fld}: {str(val).strip()}")
    return "\n".join(lines)


def render_profile(profile: dict[str, Any] | None, cfg: CCPaceConfig) -> str:
    """Render the user's long-term profile slots (history-derived, no future)."""
    if not cfg.use_profile_slots or not profile:
        return ""
    lines = []
    for key in cfg.profile_slot_keys:
        val = profile.get(key, "")
        if isinstance(val, (list, tuple)):
            val = "; ".join(str(v) for v in val)
        if str(val).strip():
            lines.append(f"{key}: {str(val).strip()}")
    return ("\nUser profile:\n" + "\n".join(lines)) if lines else ""


def render_user_block(
    history_titles: list[str],
    profile: dict[str, Any] | None,
    cfg: CCPaceConfig,
) -> str:
    hist = [t for t in history_titles[-cfg.max_history_items :] if str(t).strip()]
    hist_str = "; ".join(hist) if hist else "(no prior history)"
    return f"User recent purchases (oldest to newest): {hist_str}" + render_profile(profile, cfg)


def render_panel(
    *,
    candidates: list[dict[str, Any]],
    history_titles: list[str],
    profile: dict[str, Any] | None,
    cf_evidence: list[str] | None,
    cfg: CCPaceConfig,
    rng: random.Random,
    label_vocab: list[str] | None = None,
) -> RenderedPanel:
    """Render the full panel with a fresh label permutation.

    ``candidates`` is in ORIGINAL order (index j == candidate j). ``cf_evidence``,
    if given, is aligned to original order. The returned panel is in shuffled
    PRESENTATION order with randomized labels; ``presentation_to_original`` lets
    the caller map judge outputs back to original candidate indices.

    ``label_vocab``: optional explicit list of label strings (>= n). Defaults to
    the spec'd ``[NNN]`` bracketed ids (inference path / equivalence-tested). The
    LoRA trainer passes single-token labels so all n scores can be read from ONE
    prompt forward (the 32k-token panel only fits one forward+backward on a 4090);
    this only changes the answer tokens, not the candidate evidence or profile.
    """
    n = len(candidates)
    order = list(range(n))
    if cfg.randomize_label_ids:
        rng.shuffle(order)

    if label_vocab is not None:
        if len(label_vocab) < n:
            raise ValueError(f"label_vocab has {len(label_vocab)} < n={n} labels")
        labels = list(label_vocab[:n])
    else:
        labels = [f"[{i:03d}]" for i in range(n)]
    blocks: list[str] = []
    label_ids: list[str] = []
    original_to_label: dict[int, str] = {}
    for present_pos, orig_idx in enumerate(order):
        label = labels[present_pos]
        item = candidates[orig_idx]
        body = _fmt_item_fields(item, cfg.schema_fields)
        if cfg.use_cf_tokens and cf_evidence is not None:
            ev = str(cf_evidence[orig_idx]).strip()
            if ev:
                body += f"\ncollaborative: {ev}"
        blocks.append(f"{label}\n{body}")
        label_ids.append(label)
        original_to_label[orig_idx] = label

    return RenderedPanel(
        user_block=render_user_block(history_titles, profile, cfg),
        candidate_blocks=blocks,
        label_ids=label_ids,
        presentation_to_original=order,
        original_to_label=original_to_label,
    )


JUDGE_INSTRUCTION = (
    "You are given a user's recent purchases and {n} candidate products, each with "
    "a bracketed label. Exactly one candidate is the item the user actually bought "
    "next. Judge how strongly each candidate fits this specific user's next purchase. "
    "Answer with the single best label."
)


def build_prompt(panel: RenderedPanel, n: int) -> str:
    """Assemble the full forced-choice prompt from a rendered panel."""
    parts = [JUDGE_INSTRUCTION.format(n=n), "", panel.user_block, "", "Candidates:"]
    parts.extend(panel.candidate_blocks)
    parts.append("")
    parts.append("Best label:")
    return "\n".join(parts)
