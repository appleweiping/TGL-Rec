# TGL-Rec Redesign — Tri-Agent Design Decision (ARIS)

**Date:** 2026-06-07
**Participants:** Opus 4.8 lead (proposal A) · Opus 4.8 #2 (proposal B + critique of C) · GPT-5.5 xhigh via relay (proposal C + critique of A/B)
**Status:** converged. Method renamed **RW-PMI** (Residual-Witness-Calibrated Set-PMI). Old TGL-Rec temporal-graph core is **abandoned** (unanimous verdict).

---

## 1. What every participant agreed on
- **Abandon the TDIG / 26-param need-gate / graph-to-prose core.** It fails for a structural reason (near-zero transition edges on sparse Amazon data → evidence vector ≈ 0 → falls below popularity), not an implementation bug. Keep only the harness: 101-candidate protocol, LoRA pipeline, eval/provenance discipline.
- **The winning primitive is the backbone's own SET-NORMALIZED density ratio (PMI).** Score a candidate by `log p_θ(c|H)` debiased by its likelihood **against the exact 101-candidate set**, trained with **InfoNCE over that set**.
- **Falsify zero-shot first.** The whole premise must clear naive popularity on beauty with a frozen model before any GPU is spent on LoRA. This is the explicit antidote to the project's prior mistake (building an unvalidated premise).

## 2. Decisive technical corrections from the adversarial round
These changed the ranking of the proposals and are now baked in:

1. **A's `−log Z_set(c)` is rank-invariant** (per-user logsumexp constant) → does NOT debias popularity within a user's list. **B's per-candidate null-user marginal `log p̃(c|C_101)` is the correct, within-list debiasing term.** → adopt B's form, drop A's.
2. **A's backward term `log p(i_k|H_<k,c)` is ~zero-gradient under a causal LM** (c sits after position k, masked out); if c is prepended instead it collapses into forward similarity. → drop A's backward term as specified.
3. **C's explanatory-gain ≈ a learned-bottleneck MI that DROPS the set-marginal** (baselines against ∅, not against the 101-set) → C alone is a more expensive reparametrization missing the one term that fights popularity. → C's witnesses survive only as a residual add-on, not the core.
4. **C's one genuinely valuable idea = the PROSPECTIVE / intent-explanation framing** (does c advance a future use-case), which A/B's retrodictive PMI lacks — decisive on beauty where 10 candidates all look like "beauty routine" items and only intent-delta separates them. → keep it, but leakage-free.
5. **InfoNCE is hostage to the negatives.** With random/easy negatives the model relearns popularity. → **popularity-matched negatives are mandatory** (both reviewers, independently).
6. **Two residual threats on Amazon Beauty (GPT-5.5):** (a) the set-marginal may be *non-separable* — category/brand prior is partly real purchase signal, so over-subtraction can hurt; (b) **format-likelihood bias** — well-formatted titles get systematically higher likelihood. → the zero-shot test is exactly what detects these; add a token-length penalty `−δ·log(1+toklen(c))`.

## 3. Final method — RW-PMI

**Falsifiable claim:** In beauty 101-candidate reranking, the strongest verifiable core is set-PMI; the only CEG component worth keeping is a low-dimensional, frozen, **residualized** intent-witness gain that must provide independent ranking lift *after controlling for set-PMI*. If the witness residual adds < 0.007 NDCG@10, it is decorative and is removed.

**Scoring function:**
```
s(u,c) = [ ℓ_θ(c|H_u) − m_θ(c|C_u) ]            # set-PMI  (B's spine — real within-list debiasing)
       + α · r̃(u,c)                              # residualized witness gain  (C's kernel, leakage-free)
       − δ · log(1 + toklen(c))                  # format/length bias guard
```
- `ℓ_θ(c|H)` = length-normalized candidate continuation likelihood.
- `m_θ(c|C)` = null-user likelihood of `c` normalized **inside the 101-set** (not a bare global marginal).
- `W_u` = 4 short witnesses (category / routine-step / concern / brand-affinity-or-format) extracted by a **frozen** extractor from **recent history only** (NO future window → no leakage).
- raw witness gain `r(u,c) = g_θ(W_u|H_u,c) − g_θ(W_u|H_u,∅)`, then **residualized vs PMI**:
  `r̃(u,c) = r(u,c) − â·PMI(u,c) − b̂`  → guarantees it is not the forward term re-skinned.

**Training (LoRA on Qwen3-8B, only if zero-shot passes):**
```
L = InfoNCE_over_101(s(u,c+)) + λ·L_wit + μ·L_resid
```
- `L_wit`: candidate must explain the **frozen history-witness** (not generate a future witness).
- `L_resid = corr(r, PMI)²` in-batch → penalizes witness/PMI collinearity.
- **popularity-matched negatives**, LoRA rank 16, max history 20, full 101 rerank.

## 4. Beauty kill test (<1 day, frozen model, NO training)
Score 973 users × 101 candidates four ways: popularity, conditional-only, set-PMI, RW-PMI.
| Decision | Condition |
|---|---|
| **KILL whole premise** | set-PMI NDCG@5 < popularity (~0.104), OR set-PMI − conditional-only < +0.010 NDCG@5 |
| **KILL witness add-on only** | RW-PMI − set-PMI < +0.007 NDCG@10 AND < +0.004 MRR, OR corr(r,PMI) > 0.80 after residualization |
| **GO to LoRA** | set-PMI NDCG@5 ≥ 0.125 AND witness adds ≥ +0.007 NDCG@10 |
| **STRONG GO** | RW-PMI zero-shot NDCG@5 ≥ 0.135, NDCG@10 ≥ 0.175, MRR ≥ best frozen baseline + 0.010 |
| **Post-LoRA reportable (beauty)** | NDCG@10 ≥ 0.205, NDCG@5 ≥ 0.150, MRR ≥ baseline + 0.015, across 3 seeds std ≤ 0.006 |

Anchors: prior temporal method NDCG@5 ≈ 0.065; popularity ≈ 0.104; strong official baselines NDCG@10 ≈ 0.18.

## 5. Lead's score & ruling (Opus 总舵)
- **Novelty / defensibility:** 8/10. "Set-normalized density-ratio reranking with a residualized prospective-intent term, backbone tuned as a calibrated ratio estimator." The reviewer rebuttal ("just PMI") is answered by the set-conditioned normalization (no baseline's architecture expresses it) + the residual-independence constraint on the intent term.
- **Falsifiability / risk profile:** 9/10. Zero-shot kill test removes the project's prior failure mode (premise never tested). A dead idea dies in an afternoon, no GPU wasted.
- **Honest caveat carried forward:** GPT-5.5's non-separability concern is real — set-PMI might NOT beat popularity zero-shot on beauty. That is acceptable: the test is designed to surface it cheaply, and if it fails we learn the true blocker instead of shipping another empty SOTA claim.
- **Ruling:** proceed to implement the **zero-shot RW-PMI kill test on beauty** as the immediate next step. Do not train until set-PMI clears popularity. This passes the ARIS ≥8/10 design gate (8.3/10 mean across novelty/falsifiability/feasibility).

## 6. Immediate next action
Implement the frozen-model beauty scorer (popularity / conditional / set-PMI / RW-PMI) on the server, run the <1-day kill test, report against the table above. Needs beauty train-history + 101-candidate test task from the server (same data the baselines used).
