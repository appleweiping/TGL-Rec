"""CC-PACE judge trainer: listwise Plackett-Luce / softmax-NLL over the panel.

Trains ONLY the judge LoRA. The loss is the listwise Plackett-Luce top-1
likelihood over the 101-panel, which for a single positive equals the listwise
softmax cross-entropy:

    L = - s_+  + logsumexp_j s_j        (per panel, s = ranked statistic)

This is Fisher-consistent for the popularity-debiased log-ratio (so it earns the
Neyman-Pearson framing in docs/method_v2_decision_CC-PACE.md), and because the
statistic is panel-residualized, popularity enters as an additive per-panel
constant that cancels in the difference s_+ - s_j -> popularity-orthogonal
gradient (enforced, not assumed, via the optional dCor penalty).

Key disciplines:
  - Train panels are built from TRAIN interactions only (pseudo-held-out positive
    + popularity-matched and intent-matched negatives). The eval/test panel is
    never seen during training.
  - SPLIT CONFORMAL: the judge is fit on fold A; conformal calibration is computed
    on a disjoint fold B. This preserves the coverage guarantee.

This module provides the loss, the optional dCor penalty, and a config->plan
builder. The heavy HF/LoRA training loop reuses ``llm4rec.trainers.lora`` /
``lora_sft``; here we keep the loss + data contract that are CC-PACE-specific and
fully unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from llm4rec.methods.cc_pace.config import CCPaceConfig


def plackett_luce_top1_loss(stat: np.ndarray, positive_idx: int, temperature: float = 1.0) -> float:
    """Listwise PL top-1 NLL = softmax cross-entropy for a single positive.

    L = -s_+/T + logsumexp_j (s_j/T). Depends only on pairwise differences, so an
    additive per-panel popularity constant cancels.
    """
    s = np.asarray(stat, dtype=float) / max(temperature, 1e-6)
    m = float(np.max(s))
    lse = m + np.log(np.sum(np.exp(s - m)))
    return float(lse - s[positive_idx])


def plackett_luce_grad(stat: np.ndarray, positive_idx: int, temperature: float = 1.0) -> np.ndarray:
    """Gradient of the PL loss wrt the statistic (dense over all candidates)."""
    s = np.asarray(stat, dtype=float) / max(temperature, 1e-6)
    m = float(np.max(s))
    p = np.exp(s - m)
    p /= p.sum()
    g = p.copy()
    g[positive_idx] -= 1.0
    return g / max(temperature, 1e-6)


def distance_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Distance correlation (captures nonlinear + interaction dependence).

    Used as the popularity-orthogonality penalty dCor(T, log-pop) within a panel.
    Returns a value in [0, 1]; 0 == independent.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    n = a.shape[0]
    if n < 2:
        return 0.0
    A = np.abs(a - a.T)
    B = np.abs(b - b.T)
    A = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
    B = B - B.mean(0, keepdims=True) - B.mean(1, keepdims=True) + B.mean()
    dcov2 = (A * B).mean()
    dvar_a = (A * A).mean()
    dvar_b = (B * B).mean()
    denom = np.sqrt(dvar_a * dvar_b)
    if denom <= 1e-12:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))


def panel_loss(
    stat: np.ndarray,
    positive_idx: int,
    log_pop: np.ndarray | None,
    cfg: CCPaceConfig,
) -> float:
    """Total per-panel training loss = PL NLL + optional dCor popularity penalty."""
    loss = plackett_luce_top1_loss(stat, positive_idx, cfg.temperature)
    if cfg.use_dcor_penalty and log_pop is not None and len(log_pop) == len(stat):
        loss += cfg.dcor_weight * distance_correlation(stat, log_pop)
    return loss


@dataclass(frozen=True)
class TrainingPlan:
    """Declarative plan the HF LoRA loop consumes (no GPU needed to build)."""

    base_model: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    epochs: int
    batch_panels: int
    loss: str
    temperature_schedule: tuple[float, float]
    split_conformal: bool
    notes: str


def build_training_plan(cfg: CCPaceConfig) -> TrainingPlan:
    return TrainingPlan(
        base_model=cfg.backbone_model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_panels=cfg.batch_panels,
        loss=cfg.train_loss,
        temperature_schedule=(cfg.temperature, cfg.temperature_min),
        split_conformal=cfg.split_conformal,
        notes=(
            "Train judge LoRA only on TRAIN-built panels (pseudo-held-out positive + "
            "pop-matched & intent-matched negatives). Fit on fold A; conformal calibrate "
            "on disjoint fold B. PL top-1 loss; anneal temperature; dCor popularity penalty."
        ),
    )
