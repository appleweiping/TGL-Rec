"""Adaptive duel scheduler for PaRC.

Decides WHICH pairs to compare. Two modes:

  - ``adaptive`` (default, O(K log K)): anchored at the pony order (never a cold
    start), it runs a merge-sort-style pass whose comparisons naturally
    concentrate near already-close items, then a dueling-bandit-style boundary
    refinement that spends extra comparisons around the top-k boundary (rank
    ``boundary_k``), where NDCG@k errors actually cost. Budget-capped at
    ``duel_budget_factor * K * log2(K)`` comparisons.

  - ``full`` (O(K^2)): every unordered pair once. ONLY for the phenomenon
    sub-sample (the plan caps this at a 300-user sub-sample); the 10k-user
    ranking uses ``adaptive`` exclusively.

The scheduler is LLM-agnostic: it yields pairs ``(i, j)`` of ORIGINAL candidate
indices and a callable evaluates them (via the symmetrized duel). It returns the
list of executed pairs with their ``s_ij`` plus the count of comparisons used, so
budget compliance is auditable.

Determinism: all randomness flows through a single ``numpy.random.Generator``
seeded by the caller (the plan seeds duel scheduling, not the LLM, which is
greedy/deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from llm4rec.methods.parc.config import PaRCConfig

# A duel evaluator: given (i, j) original indices -> symmetrized s_ij (i over j).
DuelEvalFn = Callable[[int, int], float]


@dataclass
class DuelSchedule:
    """Executed duels + audit counters."""

    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (i, j, s_ij)
    n_comparisons: int = 0   # number of symmetrized duels actually evaluated
    mode: str = "adaptive"
    budget: int = 0

    def as_pair_list(self) -> list[tuple[int, int, float]]:
        return list(self.pairs)


def budget_for(n_items: int, cfg: PaRCConfig) -> int:
    """Max symmetrized comparisons for the adaptive schedule (O(K log K))."""
    if n_items <= 1:
        return 0
    return int(np.ceil(cfg.duel_budget_factor * n_items * np.log2(max(n_items, 2))))


def schedule_full(
    n_items: int,
    eval_fn: DuelEvalFn,
) -> DuelSchedule:
    """O(K^2): all unordered pairs once (phenomenon sub-sample only)."""
    sched = DuelSchedule(mode="full", budget=n_items * (n_items - 1) // 2)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            s = float(eval_fn(i, j))
            sched.pairs.append((i, j, s))
            sched.n_comparisons += 1
    return sched


def schedule_adaptive(
    pony_order: list[int],
    eval_fn: DuelEvalFn,
    cfg: PaRCConfig,
    rng: np.random.Generator,
) -> DuelSchedule:
    """O(K log K) adaptive schedule anchored at ``pony_order``.

    ``pony_order``: candidate ORIGINAL indices sorted best->worst by the pony
    posterior. We do NOT cold-start; the pony order is the prior and every
    comparison refines it locally.

    Stages:
      1. Adjacent-rank sweep over the pony order (K-1 comparisons): compares each
         neighbour pair. These are the genuinely close calls the pointwise order
         is least sure about.
      2. Boundary refinement (dueling-bandit style): around rank ``boundary_k``
         (the top-k cutoff), within a +/- ``boundary_window`` band, run
         ``refine_rounds`` extra passes comparing items across the cutoff so the
         exact top-k membership is resolved. Comparisons are deduplicated and the
         total is capped at ``budget_for``.

    Returns executed (i, j, s_ij) with i,j ORIGINAL indices (i = earlier in pony
    order within each emitted pair, for stable dedup keys).
    """
    n = len(pony_order)
    budget = budget_for(n, cfg)
    sched = DuelSchedule(mode="adaptive", budget=budget)
    if n <= 1:
        return sched

    seen: set[tuple[int, int]] = set()

    def _do(a_rank: int, b_rank: int) -> None:
        """Compare items at pony-ranks a_rank < b_rank (dedup + budget guard)."""
        if sched.n_comparisons >= budget:
            return
        i = pony_order[a_rank]
        j = pony_order[b_rank]
        key = (a_rank, b_rank) if a_rank < b_rank else (b_rank, a_rank)
        if key in seen:
            return
        seen.add(key)
        # emit with the better-ranked (earlier in pony order) item first
        lo, hi = (a_rank, b_rank) if a_rank < b_rank else (b_rank, a_rank)
        s = float(eval_fn(pony_order[lo], pony_order[hi]))
        sched.pairs.append((pony_order[lo], pony_order[hi], s))
        sched.n_comparisons += 1

    # --- Stage 1: adjacent-rank sweep over the pony order ---
    for r in range(n - 1):
        _do(r, r + 1)

    # --- Stage 2: boundary refinement around the top-k cutoff ---
    k = min(max(cfg.boundary_k, 1), n - 1)
    w = max(cfg.boundary_window, 1)
    lo = max(0, k - w)
    hi = min(n - 1, k + w)
    band = list(range(lo, hi + 1))
    for _ in range(max(cfg.refine_rounds, 0)):
        if sched.n_comparisons >= budget:
            break
        # compare items across the cutoff (one side just-above k, one just-below)
        above = [r for r in band if r < k]
        below = [r for r in band if r >= k]
        rng.shuffle(above)
        rng.shuffle(below)
        for ra, rb in zip(above, below):
            _do(ra, rb)
        # plus a few random within-band pairs to break near-ties the merge missed
        if len(band) >= 2:
            extra = min(w, max(0, budget - sched.n_comparisons))
            for _ in range(extra):
                ra, rb = rng.choice(band, size=2, replace=False)
                if ra != rb:
                    _do(int(ra), int(rb))

    return sched


def run_duels(
    pony_scores: np.ndarray,
    eval_fn: DuelEvalFn,
    cfg: PaRCConfig,
    *,
    seed: int = 0,
) -> DuelSchedule:
    """Top-level entry: pick the schedule by ``cfg.duel_mode`` and run it.

    ``pony_scores``: per-candidate pony posterior (defines the anchor order).
    ``eval_fn(i, j)``: returns symmetrized s_ij for ORIGINAL indices i, j.
    """
    n = len(pony_scores)
    if cfg.duel_mode == "full":
        return schedule_full(n, eval_fn)
    pony_order = sorted(range(n), key=lambda i: (-float(pony_scores[i]), i))
    rng = np.random.default_rng(seed)
    return schedule_adaptive(pony_order, eval_fn, cfg, rng)
