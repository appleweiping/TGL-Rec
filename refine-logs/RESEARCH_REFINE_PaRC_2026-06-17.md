# TGL-Rec ARIS research-refine — REDESIGN to PaRC (2026-06-17)

**Status:** research-refine artifact (Opus 4.8 design proposal; awaiting Codex GPT-5.5 xhigh cross-model review gate, ≥7 novelty+feasibility).
**Why a redesign:** CC-PACE failed — beauty zero-shot NDCG@10=0.1108 and LoRA-trained ≈0.086, both BELOW the ProEx bar 0.1506; and it's slow (listwise 32k-token HF judge ~16h/973, vLLM-non-viable). Recorded negative result: agentmemory mem_mqi17feh.

## Root causes of CC-PACE failure
- RC1: the 32k-token **listwise panel** is an attention/credit-assignment sink — diffuse position-contaminated logits over 101 near-homogeneous candidates; the forced-choice head has no clean per-candidate gradient. LoRA made it worse (panel-CE memorizes label distribution, not relevance; the single-token label surrogate `[017]` also tokenizes to multiple tokens on Qwen3-8B → degenerate target).
- RC2: residualization is bolted onto an unreliable base statistic.
- RC3: one giant prompt per user ⇒ batch≈1 ⇒ defeats vLLM continuous batching (intrinsic, not a bug).

## The open lane (distinct from pony)
pony-rec owns the **pointwise** posterior (absolute P(relevant|h,c), ranked raw) — it cannot see any **between-candidate / comparative** structure. CC-PACE tried to capture comparison but via the wrong vehicle (panel). The under-served, novel lane: a **cheap, vLLM-batchable, theory-grounded estimator of comparative relevance** that is provably orthogonal to the pointwise posterior.

## RECOMMENDED METHOD — PaRC (Pairwise-Relational Calibration)
- **Pairwise prompt** (short, ~300–600 tok, vLLM-batchable): "given history H, is item A or B more likely the next interaction?" → symmetrized logit `s_ij = ½(logit(A,B) − logit(B,A))` (also yields a free position-bias diagnostic).
- **Low-rank Bradley–Terry field:** `s_ij ≈ θ_i − θ_j + ε_ij`. Decompose `θ_i = α·(pony pointwise posterior_i) + β_i`. The **headline object is β** — the comparative correction the LLM makes only when forced to compare (items rated equal in isolation but consistently ordered head-to-head). β is, by construction, the part of the order **invisible to any pointwise scorer**.
- **O(K log K) adaptive duels:** anchor θ at pony's posterior (never cold, never re-derive pony); a dueling-bandit / merge-sort schedule concentrates comparisons near the top-k boundary (where NDCG@10 errors actually cost). ~600–1300 short prompts/user — fewer total tokens than CC-PACE, fully vLLM-batched.
- **Floor = pony's score** (via the α-anchor), so it clears the per-domain bar in most domains; the claim is the right inequality: anchor + comparative correction ≥ anchor.

## Research question / headline claim
Does a frozen LLM's forced pairwise comparison carry next-item ranking information **provably absent** from its pointwise posterior (measurable as significant **intransitivity** / non-BT residual), and can that comparative residual be reconstructed from O(K log K) cheap duels to improve top-k ranking at fixed compute? Headline: the intransitivity residual β is orthogonal to the pointwise posterior and lifts NDCG@10 at the top-k boundary, at ≤ CC-PACE cost.

## Experiment-plan sketch (gates)
1. **Phenomenon / KILL-GATE (1 domain, beauty, run FIRST):** compute pony pointwise order + symmetrized full pairwise order on the 101-cand panels; measure Kendall-τ gap, Var(β), and statistically-significant **cyclic intransitivity**. **Proceed only if** pairwise order beats pointwise NDCG@10 by ≥+0.005 abs AND β explains ≥10% of order variance AND intransitivity is significant. Else kill/pivot.
2. Ablation: BT-only θ vs α·pony+β; full O(K²) vs adaptive O(K log K) (≤2% NDCG loss at ≤15% comparisons); symmetrized vs one-directional (position bias).
3. Comparison: 8 domains, 10k users, Qwen3-8B, 8 official baselines, paired Holm-bootstrap; target beat strongest baseline (beauty ≥0.1506; others LLMEmb 0.094–0.274) p<0.05 in ≥6/8.
4. Mechanism: β concentrates on near-tie pairs; gains localize there; wall-clock ≤ CC-PACE, fewer tokens, same 1×RTX4090 vLLM.

## Kill-argument (strongest reviewer objection) + answerability
"A frozen LLM asked 'A or B' just computes two absolute scores and subtracts → its pairwise prefs are BT-transitive → there exists a pointwise utility reproducing the order → PaRC = a better-calibrated pony at O(K log K)× cost. You reinvented the sibling project."
**Answerable ONLY empirically, via Phase-1:** if symmetrized pairwise prefs show statistically-significant **cyclic intransitivity** (beyond BT noise), then by the BT representation theorem NO pointwise utility reproduces the order ⇒ the comparative signal is provably not in any pointwise posterior (LLM pairwise judgments are known to be order/context-sensitive and often intransitive — real prior reason to expect this). If intransitivity is negligible, the objection stands → KILL at the gate before any 8-domain compute. This binary, cheap, up-front test is the project's honest make-or-break.

## Next ARIS steps
- Codex GPT-5.5 xhigh cross-model review of this proposal (novelty + feasibility ≥7) — record verdict here.
- If pass → experiment-plan (formalize the Phase-1 kill-gate as the first milestone) → experiment-bridge (implement pairwise vLLM scorer + BT MLE + adaptive duels) → run Phase-1 kill-gate on beauty (GPU, queued behind pony) → gate decision.
