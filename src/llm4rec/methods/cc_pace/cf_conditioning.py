"""Collaborative conditioning for CC-PACE (frozen CF as the conditioning sigma-field).

CF is NOT a score head. A frozen collaborative model (e.g. SASRec) supplies, per
candidate, two distinct things in two distinct roles:
  1. EVIDENCE TOKENS  -> rendered into the judge prompt (schema.py). These let the
     judge form CF x content x intent interaction evidence. Nonlinear role.
  2. NUISANCE COORDINATES (r_cf score, cf_emb summary, cf_cluster, log-pop) ->
     fed to the residualizer (residualizer.py), which removes only their ADDITIVE
     monotone shadow. Additive role.

These roles do not cancel: subtracting the additive CF shadow (role 2) leaves the
interaction reasoning (role 1) that an additive model cannot represent -- which is
exactly CC-PACE's contribution. Ablating CF entirely (use_cf_tokens=False AND
use_cf_nuisance=False) degrades to text-only PACE with no code-path change.

This module defines the CFProvider protocol so any frozen CF backend plugs in.
``PrecomputedCFProvider`` reads CF artifacts produced offline by an existing
baseline runner (e.g. the SASRec ranker already in this repo), so CC-PACE never
trains collaborative parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class CFProvider(Protocol):
    """Supplies frozen collaborative signal for one user's candidate panel."""

    def evidence_tokens(self, user_id: str, candidate_ids: list[str]) -> list[str]:
        """Rendered neighbour evidence string per candidate (original order)."""

    def nuisance(self, user_id: str, candidate_ids: list[str]) -> dict[str, np.ndarray]:
        """Nuisance coordinates per candidate: r_cf, cf_emb (scalar summary), cf_cluster."""


@dataclass
class PrecomputedCFProvider:
    """Frozen CF signal loaded from offline artifacts.

    ``scores[user_id][item_id]`` -> CF relevance score (e.g. SASRec logit).
    ``neighbors[user_id][item_id]`` -> short list of co-purchase/neighbour titles.
    ``clusters[item_id]`` -> integer cluster id in CF-embedding space.
    Anything missing degrades gracefully (empty evidence / zero nuisance).
    """

    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    neighbors: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    clusters: dict[str, int] = field(default_factory=dict)

    def evidence_tokens(self, user_id: str, candidate_ids: list[str]) -> list[str]:
        u_nbr = self.neighbors.get(user_id, {})
        u_sc = self.scores.get(user_id, {})
        out: list[str] = []
        for cid in candidate_ids:
            nbrs = u_nbr.get(cid, [])
            rank_hint = u_sc.get(cid)
            parts = []
            if nbrs:
                parts.append("similar users who bought this also bought: " + "; ".join(nbrs[:5]))
            if rank_hint is not None:
                parts.append(f"collaborative affinity score={float(rank_hint):.3f}")
            out.append(" | ".join(parts))
        return out

    def nuisance(self, user_id: str, candidate_ids: list[str]) -> dict[str, np.ndarray]:
        u_sc = self.scores.get(user_id, {})
        r_cf = np.array([float(u_sc.get(cid, 0.0)) for cid in candidate_ids])
        cf_cluster = np.array([float(self.clusters.get(cid, 0)) for cid in candidate_ids])
        return {"r_cf": r_cf, "cf_cluster": cf_cluster, "cf_emb": r_cf.copy()}


@dataclass
class NullCFProvider:
    """No collaborative signal (text-only PACE ablation)."""

    def evidence_tokens(self, user_id: str, candidate_ids: list[str]) -> list[str]:
        return ["" for _ in candidate_ids]

    def nuisance(self, user_id: str, candidate_ids: list[str]) -> dict[str, np.ndarray]:
        return {}
