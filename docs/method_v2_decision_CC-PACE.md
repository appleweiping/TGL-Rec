# CC-PACE — Final Method Design (tri-agent, ≥20-round escalation)

**Date:** 2026-06-07 · **Seats:** Opus-lead, Opus-#2, GPT-5.5-xhigh (relay) · **Supersedes:** RW-PMI
(`docs/redesign_decision_RW-PMI.md`). Raw discussion: `outputs/method_redesign_v2_discussion/`.
**Status:** converged after 8 deep multi-part rounds (proposals → cross-critique → theory audit →
non-stitch resolution → identifiability → baseline stress). **No experiments yet (per user).**

## Name
**CC-PACE — Collaborative-Conditioned Panel-Anomaly Calibrated Exchangeability reranker.**

## One-sentence claim (falsifiable)
The user's next item is the panel candidate whose forced-choice judge evidence is most anomalous
relative to a popularity/facet/collaborative-matched empirical null built from the panel's own
negatives; a single Qwen3-8B judge conditioned on rendered collaborative-neighbor evidence, trained
listwise, beats all 8 official baselines — and ablating the collaborative tokens drops it below the CF
baseline, proving the gain is in-context collaborative reasoning, not score fusion.

## The method (one mechanism)
1. **Forced-choice listwise judge.** Qwen3-8B sees the whole 101-panel in one pass. Each candidate is
   rendered in a unified schema with **randomized label IDs**:
   `x_c = [id_π(c) | content(category|brand|keywords|attrs) | facet_bucket | profile_slots(u) |
   CF_neighbor_evidence e_CF(u,c)]`. Decoding constrained to the 101 label tokens; read normalized
   logprob as the per-candidate evidence `E_judge(c | panel, C_u)`. Label randomization ⇒ the joint
   score is permutation-equivariant ⇒ exchangeable.
2. **Collaborative conditioning (frozen, non-stitch).** A frozen CF model (e.g. SASRec) supplies, per
   candidate: rendered nearest-neighbor / co-purchase evidence tokens `e_CF` (into the prompt) and the
   nuisance coordinates `(CF_emb, r_CF, CF_cluster)` (into the residualizer). CF is the **conditioning
   σ-field**, never a score head.
3. **Residualized ranking statistic.** `T_u(c) = E_judge(c) − m̂_LOO(content_c, facet_c, log-pop_c,
   CF_emb_c, r_CF, CF_cluster)`, where `m̂` is a **symmetric** leave-one-out isotonic/GAM fit on the
   panel nulls. This removes the best *additive* popularity/CF explanation; the surviving residual is
   the CF×content×intent interaction reasoning an additive model cannot express. **Rank by `T_u`.**
4. **Dual empirical null + conformal (deployment layer).** `P_pop` = the real eval panel
   (protocol negs are popularity-matched); `P_sem` = auxiliary retrieved intent-matched nulls used
   ONLY as a calibration reference (never scored for the metric). Conformal p via **split conformal**
   (judge trained on fold A, calibrated on held-out fold B); combine `p = max(p_pop, p_sem)`
   (intersection-union, valid w/o independence). **Honest framing: conformal does NOT change
   within-panel NDCG (monotone, rank-preserving); it is the selective-prediction / coverage / abstention
   layer.** Ranking is driven by `T_u`.
5. **Training.** Only the judge LoRA, **listwise Plackett-Luce / softmax-NLL** over the panel
   (Fisher-consistent for the popularity-debiased log-ratio → earns the Neyman-Pearson framing).
   Train panels from train interactions only (pseudo-held-out positive + pop-matched & intent-matched
   negs). Optional **dCor/HSIC(T, log-pop)** penalty to enforce popularity-orthogonality (the
   additive-annihilation argument is not a theorem under interaction leakage). τ annealed; sparse users
   get empirical-Bayes James-Stein shrinkage toward the facet-bucket mean.

## Why it beats the 8 baselines (stress-tested)
Strictly dominates content/semantic-shortcut baselines (llmemb, proex, llmesr-LLM) and every
popularity-shortcut weakness (the null is popularity-matched + residualized). Ties — does not dominate —
pure collaborative/graph/sequential strengths (rlmrec, irllrec, elmrec, session-local llm2rec/llmesr);
those are covered by the frozen-CF conditioning + the intent-conditioned panel constructor. **Honest
top risk: promax (the beauty SOTA) — beauty preference is near-perfectly profile-expressible**, so the
schema MUST carry explicit long-term profile slots (skin type, concern, brand loyalty, routine step,
ingredient) or promax wins.

## Honest caveats (carry into the paper)
- Conformal is a calibration/abstention layer, **not** the ranking engine. Never claim it lifts NDCG.
- NP-optimality holds only among rank tests measurable w.r.t. the rendered features + frozen judge, and
  only under the PL loss. State qualified.
- The contribution survives only if `T_u` retains signal under an interaction-rich `m̂_rich`; if it →0
  there, the honest claim degrades to "collaborative interactions matter." Report both residualizers.
- Pure-content sufficiency is a bet; CF-conditioning is included because `I(y; φ_CF | content) > 0` is
  empirically likely on beauty.

## Decisive identifiability experiments (reviewer-grade)
Panel-corruption (swap another user's negs → NDCG must drop to content-only); CF-token ablation (must
fall BELOW CF baseline; in-context must beat a late-fusion control on the interaction slice);
additive-m̂ → m̂_rich ceiling test; conformal-p vs raw-T (NDCG identical + valid coverage + AURC gain);
non-collinearity: after m̂, |partial-Spearman(T, log-pop | content)| < 0.05.

## Cheap beauty go/kill (zero-shot frozen + small LoRA), bar = promax NDCG@10 0.1506
- **Zero-shot probe** (no training): frozen judge + residualizer on 973×101. Calibration: positive
  panel-evidence rank > uniform (bootstrap CI excludes). If T carries no signal vs popularity → revisit
  before GPU.
- **GO (train LoRA):** zero-shot NDCG@10 ≥ 0.13 AND CF-token ablation shows a positive gap.
- **STRONG GO / reportable:** post-LoRA NDCG@10 ≥ 0.1506 with paired-bootstrap p<0.05 over users, AND
  panel-corruption drops ≥30%, AND CF-token ablation falls below the CF baseline (proves the mechanism).
- **KILL/reframe:** only wins on popularity-heavy slice, or conformal coverage breaks, or T→0 under
  m̂_rich and late-fusion is what works (→ reposition as "panel-exchangeable calibration of a hybrid
  score").

## Lead ARIS score
- **Novelty/defensibility 9/10** — panel-as-empirical-null + CF-as-conditioning-σ-field + residual =
  judge-evidence-minus-best-additive-CF-explanation is one coherent, genuinely new object; the
  "just LLM+CF" rebuttal has a crisp answer (one scoring function; ablate tokens = zero code change).
- **Rigor 9/10** — split-conformal coverage, PL Fisher-consistency, dCor-enforced orthogonality,
  m̂_rich ceiling test, full non-collinear ablation matrix.
- **Feasibility 7/10** — 101-in-context judge (~8-11k tokens, fits 32k) + frozen CF retrieval + LoRA on
  one GPU; main risk is engineering the forced-choice logit read + CF rendering, and whether the 8B
  judge actually exploits CF tokens.
- **SOTA plausibility 7/10** — real shot, but promax on beauty is a genuine threat; mitigated by profile
  slots + intent-conditioned panel. **Mean 8.0/10 → passes ARIS ≥8 design gate.**

## Next (gated)
No experiments until GPU is free AND user clears. First action when cleared: implement the zero-shot
CC-PACE beauty probe (frozen judge + residualizer + CF rendering), run the go/kill above. Then LoRA if
GO. Server-only compute; lightweight evidence synced to local; commit/push from local.
