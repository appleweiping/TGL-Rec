"""Configuration for PaRC (Pairwise-Relational Calibration).

A single frozen dataclass captures every knob the method exposes, so configs are
reproducible and every hyperparameter is sweepable for the paper's
hyperparameter-stability + ablation experiments (see
``refine-logs/EXPERIMENT_PLAN_PaRC.md`` Block 1/4).

PaRC anchors on the pony pointwise posterior (``pony_i``) and adds a
validation-gated comparative correction ``beta_i`` reconstructed from O(K log K)
adaptive pairwise duels. Final score is ``score_i = pony_i + lambda * beta_i``
with ``lambda >= 0`` selected on a validation split (``lambda = 0`` allowed ->
floor recovers pony). The ``use_*`` switches let the ablation harness disable
each component independently (symmetrization, anchoring, adaptive scheduling).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaRCConfig:
    """All PaRC hyperparameters and structural switches."""

    # --- identity ---
    name: str = "parc"
    codename: str = "PaRC"
    eval_split: str = "test"

    # --- pairwise prompt ---
    max_history_items: int = 5            # short prompt: ~300-600 tok target
    title_max_chars: int = 80             # truncate long titles to bound prompt length
    desc_max_chars: int = 0               # 0 = titles only (keep duel prompts short)
    neutral_labels: bool = True           # use neutral "A"/"B" labels, not item names
    symmetrize: bool = True               # s_ij = 0.5*(logit(A,B) - logit(B,A))

    # --- duel scheduler (adaptive O(K log K)) ---
    duel_mode: str = "adaptive"           # {"adaptive","full"}; "full" = O(K^2) sub-sample only
    duel_budget_factor: float = 6.0       # max comparisons = factor * K * log2(K) (adaptive)
    boundary_k: int = 10                  # concentrate comparisons near top-k boundary
    boundary_window: int = 8              # +/- window around the boundary to refine
    refine_rounds: int = 2                # extra dueling-bandit refinement passes at the boundary

    # --- Bradley-Terry field (theta_i = alpha*pony_i + beta_i) ---
    bt_l2_beta: float = 1.0               # ridge on beta (sparse/partial duel graphs)
    bt_max_iter: int = 200
    bt_tol: float = 1e-7
    bt_fit_alpha: bool = True             # if False, alpha fixed to 1.0 (BT-only-residual mode)
    anchor_pony: bool = True              # if False, theta = beta only (BT-only ablation, no anchor)

    # --- calibration (validation-gated lambda mixture) ---
    lambda_grid: tuple[float, ...] = (
        0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0,
    )
    lambda_metric: str = "ndcg@10"        # validation selection objective
    standardize_beta: bool = True         # z-score beta before mixing (scale-match to pony)

    # --- intransitivity diagnostic ---
    cyclic_triples_sample: int = 0        # 0 = exhaustive; >0 = sampled triples (large K)
    null_bootstrap: int = 500             # bootstrap reps for the BT+heteroskedastic-noise null
    null_seed: int = 20260506             # frozen seed for the null bootstrap

    # --- backbone provenance (the duel callable is injected, never built here) ---
    backbone_model: str = "/home/ajifang/models/Qwen/Qwen3-8B"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PARC_CONFIG = PaRCConfig()


def load_parc_config(path_or_config: "str | Path | dict[str, Any] | None") -> PaRCConfig:
    """Build a PaRCConfig from a dict / YAML path, falling back to defaults.

    Unknown keys are ignored (forward-compatible); known keys override defaults.
    Mirrors ``llm4rec.methods.cc_pace.config.load_cc_pace_config``.
    """
    if path_or_config is None:
        return DEFAULT_PARC_CONFIG
    if isinstance(path_or_config, (str, Path)):
        from llm4rec.experiments.config import load_yaml_config

        raw = load_yaml_config(path_or_config)
        raw = raw.get("parc", raw) if isinstance(raw, dict) else {}
    else:
        raw = dict(path_or_config)
    fields = set(DEFAULT_PARC_CONFIG.to_dict())
    overrides = {k: v for k, v in raw.items() if k in fields}
    # tuples survive a YAML round-trip as lists; coerce back so the frozen dataclass
    # stays hashable / order-stable.
    for tup_key in ("lambda_grid",):
        if tup_key in overrides and isinstance(overrides[tup_key], list):
            overrides[tup_key] = tuple(overrides[tup_key])
    return PaRCConfig(**{**DEFAULT_PARC_CONFIG.to_dict(), **overrides})
