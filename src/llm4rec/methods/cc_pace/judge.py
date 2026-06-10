"""Forced-choice listwise judge for CC-PACE.

The judge sees the full label-randomized panel in ONE pass and emits a per-label
evidence scalar e_i = the model's normalized log-probability of choosing label i
as the "best" candidate. Reading per-label logits (NOT the argmax decision)
preserves the full rank distribution; combined with label randomization the joint
score is permutation-equivariant -> exchangeable.

The actual model is injected via the ``EvidenceModel`` protocol so the rest of
CC-PACE (residualizer, statistic, conformal, metrics) is unit-testable on CPU
without loading an 8B model. ``HFForcedChoiceModel`` is the concrete Qwen3-8B
implementation used on the server.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

import numpy as np

from llm4rec.methods.cc_pace.config import CCPaceConfig
from llm4rec.methods.cc_pace import schema as schema_mod


class EvidenceModel(Protocol):
    """Returns one evidence logit per label for a forced-choice prompt.

    Implementations score the next-token distribution over the panel's label
    tokens given ``prompt``, and return the (length-normalized if multi-token)
    log-probability for each label in ``label_ids`` order.
    """

    def label_logprobs(self, prompt: str, label_ids: list[str]) -> list[float]:
        ...


def judge_panel(
    *,
    candidates: list[dict[str, Any]],
    history_titles: list[str],
    profile: dict[str, Any] | None,
    cf_evidence: list[str] | None,
    model: EvidenceModel,
    cfg: CCPaceConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run R label-randomized forced-choice passes; return per-ORIGINAL-index stats.

    Returns dict with:
      - ``evidence``: mean evidence per original candidate index (np.ndarray[n])
      - ``uncertainty``: std across the R randomizations (np.ndarray[n])
    Averaging over label permutations de-noises position/label artefacts; the
    spread feeds the sparse-user shrinkage and the conformal abstention layer.
    """
    n = len(candidates)
    runs = max(1, cfg.n_label_randomizations if cfg.randomize_label_ids else 1)
    ev_runs = np.full((runs, n), np.nan)

    for r in range(runs):
        rng = random.Random(seed * 1000 + r)
        panel = schema_mod.render_panel(
            candidates=candidates,
            history_titles=history_titles,
            profile=profile,
            cf_evidence=cf_evidence,
            cfg=cfg,
            rng=rng,
        )
        prompt = schema_mod.build_prompt(panel, n)
        logps = model.label_logprobs(prompt, panel.label_ids)  # presentation order
        for present_pos, orig_idx in enumerate(panel.presentation_to_original):
            ev_runs[r, orig_idx] = float(logps[present_pos])

    evidence = np.nanmean(ev_runs, axis=0)
    uncertainty = np.nanstd(ev_runs, axis=0) if runs > 1 else np.zeros(n)
    return {"evidence": evidence, "uncertainty": uncertainty}
