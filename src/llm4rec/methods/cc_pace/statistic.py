"""CC-PACE ranked statistic: compose judge evidence -> residual -> shrink -> T_u.

This is the orchestration layer that produces the per-candidate ranking statistic
T_u and (optionally) conformal p-values. It is backbone-agnostic: it takes an
``EvidenceModel`` (judge.py) and a ``CFProvider`` (cf_conditioning.py), so it runs
on CPU in tests with a mock model and on GPU in production with HFForcedChoiceModel.

Ranking is by T_u (higher = better). Conformal p-values are attached as metadata
for the selective-prediction layer but do NOT change the order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from llm4rec.methods.cc_pace import conformal as conformal_mod
from llm4rec.methods.cc_pace import judge as judge_mod
from llm4rec.methods.cc_pace import residualizer as resid_mod
from llm4rec.methods.cc_pace.cf_conditioning import CFProvider
from llm4rec.methods.cc_pace.config import CCPaceConfig


def _log_pop(candidates: list[dict[str, Any]]) -> np.ndarray:
    pops = np.array([float(c.get("popularity", c.get("pop_count", 0.0))) for c in candidates])
    return np.log1p(np.clip(pops, 0.0, None))


def _facet_bucket(candidates: list[dict[str, Any]]) -> np.ndarray:
    cats = [str(c.get("category", "")).split(">")[0].strip() for c in candidates]
    uniq = {c: i for i, c in enumerate(sorted(set(cats)))}
    return np.array([uniq[c] for c in cats], dtype=float)


def compute_statistic(
    *,
    user_id: str,
    candidates: list[dict[str, Any]],
    candidate_ids: list[str],
    history_titles: list[str],
    profile: dict[str, Any] | None,
    model: judge_mod.EvidenceModel,
    cf_provider: CFProvider,
    cfg: CCPaceConfig,
    seed: int,
    n_history: int,
) -> dict[str, np.ndarray]:
    """Return per-original-index {'T','evidence','p_pop','uncertainty'} arrays."""
    n = len(candidates)

    cf_tokens = cf_provider.evidence_tokens(user_id, candidate_ids) if cfg.use_cf_tokens else None
    judged = judge_mod.judge_panel(
        candidates=candidates,
        history_titles=history_titles,
        profile=profile,
        cf_evidence=cf_tokens,
        model=model,
        cfg=cfg,
        seed=seed,
    )
    evidence = judged["evidence"]
    uncertainty = judged["uncertainty"]

    # assemble nuisance coordinates
    nuisance: dict[str, np.ndarray] = {}
    if cfg.use_residualizer:
        nuisance["log_pop"] = _log_pop(candidates)
        nuisance["facet_bucket"] = _facet_bucket(candidates)
        if cfg.use_cf_nuisance:
            nuisance.update(cf_provider.nuisance(user_id, candidate_ids))

    if cfg.use_residualizer and nuisance:
        active_keys = tuple(k for k in cfg.residual_nuisance_keys if k in nuisance)
        stat = resid_mod.residualize(
            evidence, nuisance, keys=active_keys, rich=cfg.residualizer_rich
        )
    else:
        stat = evidence.astype(float).copy()

    if cfg.use_shrinkage:
        stat = conformal_mod.james_stein_shrink(
            stat,
            uncertainty,
            nuisance.get("facet_bucket"),
            n_history=n_history,
            n_cap=cfg.shrinkage_n_cap,
        )

    p_pop = conformal_mod.all_conformal_p(stat)
    return {"T": stat, "evidence": evidence, "uncertainty": uncertainty, "p_pop": p_pop}
