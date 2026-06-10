"""CC-PACE: Collaborative-Conditioned Panel-Anomaly Calibrated Exchangeability reranker.

This package implements the method decided in
``docs/method_v2_decision_CC-PACE.md`` (tri-agent design, 2026-06-07), which
supersedes RW-PMI and the earlier TDIG/need-gate design.

ONE mechanism:
  1. A single Qwen3-8B forced-choice listwise judge sees the whole 101-candidate
     panel in one pass. Each candidate is rendered in a unified schema with a
     RANDOMIZED label id (``schema.py``), so the joint score is
     permutation-equivariant over the panel -> exchangeable.
  2. Frozen collaborative-filtering signal enters as (a) rendered neighbour
     EVIDENCE TOKENS in the prompt and (b) nuisance COORDINATES in the
     residualizer. CF is the conditioning sigma-field, never a score head
     (``cf_conditioning.py``). Dropping the CF tokens degrades CC-PACE to
     text-only PACE with zero code-path change -- the non-stitch proof.
  3. The ranked statistic is the residualized judge evidence
     ``T_u(c) = E_judge(c) - m_hat_LOO(content, facet, log-pop, CF nuisance)``
     (``residualizer.py`` + ``statistic.py``). Ranking is driven by T_u.
  4. A dual empirical null (P_pop = the eval panel; P_sem = auxiliary
     intent-matched reference) + split conformal gives a calibration /
     selective-prediction layer (``conformal.py``). HONEST: conformal does NOT
     change within-panel NDCG (monotone, rank-preserving); it is an
     abstention/coverage layer only.
  5. Training (``llm4rec.trainers.cc_pace_trainer``) fine-tunes only the judge
     LoRA with a listwise Plackett-Luce / softmax-NLL loss.

See ``CCPaceRanker`` in ``llm4rec.rankers.cc_pace`` for the BaseRanker-compatible
entry point used by the evaluation harness.
"""

from __future__ import annotations

from llm4rec.methods.cc_pace.config import CCPaceConfig, DEFAULT_CC_PACE_CONFIG

__all__ = ["CCPaceConfig", "DEFAULT_CC_PACE_CONFIG"]
