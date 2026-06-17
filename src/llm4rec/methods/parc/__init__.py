"""PaRC: Pairwise-Relational Calibration for LLM4Rec.

Anchored comparative-residual calibration. The final score is

    score_i = pony_i + lambda * beta_i        (lambda >= 0, validation-selected)

where ``pony_i`` is the EXISTING pony pointwise posterior (the alpha-anchor, never
re-derived) and ``beta_i`` is the comparative correction a frozen LLM makes only
under forced pairwise comparison, estimated by a symmetrized Bradley-Terry field
over O(K log K) adaptive, vLLM-batchable duels anchored at the pony order.
``lambda = 0`` recovers the pony order (the empirical floor); ``lambda > 0`` shifts
the ranking toward the comparative residual when validation supports it.

Design + plan: ``refine-logs/RESEARCH_REFINE_PaRC_2026-06-17.md`` (method/math) and
``refine-logs/EXPERIMENT_PLAN_PaRC.md`` (blocks, M0 gate, metrics).

Modules:
  - ``config``          -- frozen PaRCConfig (all knobs, ablation switches).
  - ``pairwise_prompt`` -- short A/B duel prompt + A/B-swap symmetrization
                           (s_ij), neutral labels, title truncation; INJECTABLE
                           ``PairwiseDuelModel`` (no vLLM/GPU at import time).
  - ``bt_field``        -- anchored symmetrized BT MLE theta = alpha*pony + beta
                           (+ Var(beta), order-variance fraction diagnostics).
  - ``duels``           -- O(K log K) adaptive merge-sort/dueling-bandit scheduler
                           anchored at the pony order (+ O(K^2) full mode).
  - ``calibrate``       -- validation-gated lambda selection + intransitivity
                           (cyclic-triple rate vs BT+heteroskedastic-noise null).
  - ``ranker``          -- PaRCRanker (BaseRanker-compatible).

The importable core has NO network/GPU dependency; the LLM duel call is injected.
"""

from __future__ import annotations

from llm4rec.methods.parc.bt_field import BTFieldResult, fit_bt_field
from llm4rec.methods.parc.calibrate import (
    IntransitivityResult,
    LambdaSelection,
    cyclic_triple_rate,
    intransitivity_test,
    mixture_score,
    ndcg_at_k,
    select_lambda,
)
from llm4rec.methods.parc.config import DEFAULT_PARC_CONFIG, PaRCConfig, load_parc_config
from llm4rec.methods.parc.duels import DuelSchedule, budget_for, run_duels, schedule_adaptive, schedule_full
from llm4rec.methods.parc.pairwise_prompt import (
    PairwiseDuelModel,
    build_pairwise_prompt,
    symmetrized_duel,
    truncate_title,
)
# NOTE: PaRCRanker is intentionally NOT imported here (mirrors cc_pace): the
# package __init__ stays lightweight so importing a leaf module (bt_field, etc.)
# never pulls in rankers.base -> rankers/__init__ -> rankers.parc, which would be
# a circular import. The ranker is reached via `llm4rec.rankers.parc` (registry
# shim) or `llm4rec.methods.parc.ranker` directly.

__all__ = [
    "PaRCConfig",
    "DEFAULT_PARC_CONFIG",
    "load_parc_config",
    "PairwiseDuelModel",
    "build_pairwise_prompt",
    "symmetrized_duel",
    "truncate_title",
    "BTFieldResult",
    "fit_bt_field",
    "DuelSchedule",
    "budget_for",
    "run_duels",
    "schedule_adaptive",
    "schedule_full",
    "LambdaSelection",
    "IntransitivityResult",
    "select_lambda",
    "mixture_score",
    "ndcg_at_k",
    "cyclic_triple_rate",
    "intransitivity_test",
]
