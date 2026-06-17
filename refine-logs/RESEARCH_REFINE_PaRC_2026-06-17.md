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

## Codex GPT-5.5 xhigh cross-review v1 (2026-06-17): NOVELTY 7/10, FEASIBILITY 6/10 → BELOW gate (both ≥7). Iterate.
Codex's binding objections: (i) "floor = pony" is FALSE without an explicit validation-gated mixture; (ii) the "significant intransitivity REQUIRED" gate is the wrong criterion (BT-transitive can still help; intransitivity can be noise/bias that hurts NDCG); (iii) "absent from ANY pointwise scorer" is overclaimed; (iv) O(K log K) over 8×10k = millions–~100M prompts — feasible only with very short prompts + strict caching + fast Phase-1 kill; (v) fatal risk: +0.005–0.010 lift at 10–100× pony cost reads as "expensive calibration layer, not a new method".

## PaRC v2 — revisions per Codex (resolves the feasibility-6 dings)
1. **Reframed claim** (fixes novelty framing): "**Anchored comparative-residual calibration for LLM4Rec**" — pony's pointwise posterior + a validation-gated comparative correction. NOT "pairwise LLM ranking" (RankGPT exists), NOT "absent from ANY pointwise scorer". Say precisely: "not captured by this pointwise posterior under the same frozen LLM + prompt family."
2. **Validation-gated mixture = real empirical floor** (the key fix): final score `score_i = pony_i + λ·β_i`, with **λ≥0 selected on validation, λ=0 allowed**. ⇒ floor = pony is now TRUE (λ→0 recovers pony); no reliance on the α-anchor for monotonicity.
3. **Two-path Phase-1 gate** (replaces "intransitivity required"): PROCEED iff PaRC beats pony by **≥+0.005 NDCG@10 with significant paired bootstrap** on a pilot domain; **separately** claim the intransitivity finding ONLY if cyclic residual is significant AND predictive of the lift. Separate three signals explicitly: BT-transitive utility improvement / β-residual-over-pony / non-BT cyclic structure.
4. **Compute-normalized evaluation** (defuses "expensive calibration"): report NDCG@10 lift **per 1k prompts and per GPU-hour** vs pony, vs RankGPT-style pairwise rerank, vs short-context listwise rerank. Headline must show strong lift OR compute-normalized value OR a scientific finding that stands even at modest gains.
5. **Aggressive position/bias controls**: symmetrized s_ij (A/B swap) + neutral labels + title-truncation + repeated-duel CIs on s_ij; report bias magnitude as a diagnostic.
6. **Pilot domain = a WINNING domain with clear margin + top-k headroom (NOT beauty** — ProEx 0.1506 high + prior LLM attempts failed there). Pick e.g. a domain where pony beats the bar comfortably so the floor clears and there's room to improve the top-k order.
7. **Pre-registered kill**: if validation λ→~0, OR β explains order-variance but yields no significant NDCG lift, KILL the method (do not push to 8 domains).

## Next ARIS steps
- Re-run Codex cross-review on PaRC v2 (expect feasibility ≥7 given the λ-floor + two-path gate + compute-normalized framing) — record verdict here. Gate: both ≥7.
- On pass → ARIS experiment-plan: formalize the two-path Phase-1 gate as milestone M0 on the chosen pilot domain; baselines incl. pony, RankGPT-pairwise, short-context listwise; pre-registered kill criteria; compute-normalized metrics. → experiment-bridge (short pairwise vLLM scorer reusing pony's run_ccrp_v3 vLLM patterns + symmetrized BT MLE + adaptive merge-sort duels + validation-gated λ) → run M0 (GPU, queued behind pony) → gate decision.
