# TGL-Rec ARIS experiment-plan — PaRC (Pairwise-Relational Calibration) — 2026-06-17

**Gate context:** research-refine PASSED (Codex GPT-5.5 xhigh 8/7; artifact `RESEARCH_REFINE_PaRC_2026-06-17.md`). Experiment-plan gate (Codex ≥6 all dims): **v1 = EVIDENCE 8 / RIGOR 7 / GATES 5 / FEASIBILITY 8 / PAPER_POTENTIAL 7** (GATES<6, blocker = M0 test-set decision leakage + kill-criteria conflict). **v2 (this version) fixes:** (1) all M0 proceed/kill + λ,α selection on validation/held-out pilot, official TEST untouched until Block 2; (2) test claim = non-inferiority vs pony (no asserted dominance); (3) method-success vs diagnostic-negative paper paths explicitly separated (no shared headline); (4) seeded sources specified (duel scheduling, λ-bootstrap, prompt perturbations — not model randomness, vLLM is deterministic); (5) intransitivity tested vs strict BT+heteroskedastic-noise null; (6) RankGPT/listwise baselines compute-matched + budget-normalized. **v2 PASSED the experiment-plan gate: Codex GPT-5.5 xhigh = EVIDENCE 8 / RIGOR 8 / GATES 7 / FEASIBILITY 7 / PAPER_POTENTIAL 8 → PASS.** Non-binding tightenings folded in: ε=0.002 abs NDCG@10 non-inferiority margin; split manifest (toys valid panel + fixed 1k held-out pilot users disjoint from test, seed 20260506); phenomenon O(K²) capped at a 300-user sub-sample (10k ranking uses only adaptive O(K log K)). → Proceed to ARIS **experiment-bridge** (code the PaRC pairwise vLLM scorer; M0 run GPU-queued behind pony).

## Headline (v2-reframed)
**Anchored comparative-residual calibration for LLM4Rec.** Final score `score_i = pony_i + λ·β_i`, λ≥0 selected on validation (λ=0 allowed ⇒ floor = pony's pointwise posterior). `β_i` is the comparative correction a frozen Qwen3-8B makes only under forced pairwise comparison, estimated via a low-rank Bradley–Terry (BT) field over O(K log K) cheap, vLLM-batchable duels anchored at pony's posterior. Question: does that comparative residual carry top-k ranking signal **not captured by this pointwise posterior (same frozen LLM + prompt family)**, lifting NDCG@10 at fixed compute? Plus a scientific sub-finding: is the LLM's pairwise preference field non-BT (significant cyclic intransitivity)?

## Frozen protocol (shared with pony/truce — non-negotiable, full-scale)
8 Amazon domains, 10k users (beauty 973), 101 same candidates/event (1 pos + 100 popularity-matched neg), Qwen3-8B, metrics HR@5/10/20 + NDCG@5/10/20 + MRR, paired Holm-corrected bootstrap. 8 official baselines frozen in `data/pony_official_baselines/`. PaRC reuses the **existing pony Qwen pointwise scores** (`outputs/<dom>_*_ccrp_v3/scores.csv`) as the α-anchor — never re-derives pony.

## Pilot domain (M0): **toys** (backup: tools)
Chosen per Codex rule "winning domain, clear margin + top-k headroom, NOT beauty": toys Qwen NDCG@10 = 0.2708 vs strongest baseline LLMEmb 0.2049 (+32% margin ⇒ floor clears comfortably) and has top-k re-ordering headroom (HR@20 0.506 ≫ NDCG@10 ⇒ relevant items present but mis-ordered). Beauty excluded (ProEx 0.1506 high; all 3 backbones underperform there).

## Experiment blocks

### Block 0 — M0 PHENOMENON + KILL-GATE (pilot=toys, run FIRST; GPU)
- **Pairwise duel scorer**: short prompt (~300–600 tok) "given history H, is A or B the more likely next interaction?"; symmetrized `s_ij = ½(logit(A,B) − logit(B,A))` (A/B-swap; also yields a position-bias diagnostic). vLLM-batched + guided-decoding (reuse pony `run_ccrp_v3` vLLM patterns).
- **BT field**: fit `θ_i = α·pony_i + β_i` by symmetrized BT MLE; select **λ (and α) on the toys VALIDATION split** for `score_i = pony_i + λ·β_i` (λ=0 allowed).
- **STRICT no-test-leakage protocol (fixes the gate blocker):** ALL proceed/kill and hyperparameter (λ,α,duel-budget) decisions are made on the **toys validation panel + a disjoint held-out pilot user subset**. SPLIT MANIFEST (committed before M0): validation = `outputs/baselines/external_tasks/toys_large10000_100neg_valid_same_candidate` (the existing valid panel); held-out pilot = a fixed **1,000-user subset drawn (seed 20260506) from the validation users, disjoint from the 10k official TEST users** → `refine-logs/parc_m0_pilot_users.csv`. The official toys **TEST panel is NOT touched at M0** — scored once, later, only inside the frozen 8-domain comparison (Block 2), so no headline statistic is post-selected.
- **Phenomenon measurements** (validation/pilot users; near-full O(K²) duels on a **capped 300-user phenomenon sub-sample**, ≈300·101²/2 ≈ 1.5M duels sampled — the 10k-user ranking uses ONLY adaptive O(K log K)): Kendall-τ(pony order, pairwise order); Var(β) / fraction of order-variance explained by β; **cyclic-intransitivity** tested against a *strict* null = BT + per-pair heteroskedastic comparison-noise (bootstrap CIs on cyclicity, not a generic BT-noise null); whether intransitivity predicts the per-event lift.
- **DECISION (validation-only), with two explicitly separated paper paths:**
  - **Method-success path → PROCEED to 8-domain:** on validation, PaRC beats pony by **≥ +0.005 NDCG@10 with significant paired bootstrap** AND λ does not collapse to ~0. On TEST (Block 2) the method claim uses a **non-inferiority margin** (PaRC ≥ pony − ε, **ε = 0.002 absolute NDCG@10, pre-set**) for the floor, plus the lift where significant — we do NOT assert guaranteed test dominance.
  - **Scientific-negative / diagnostic path:** if validation λ→~0 OR no significant validation lift, **the method is KILLED (no 8-domain scale-up)** — but the *characterized finding* about the LLM pairwise-preference field (orthogonality of β to the pointwise posterior, intransitivity structure, compute cost) is a separate, honestly-labeled diagnostic/negative contribution, NOT a "PaRC beats SOTA" claim. The two paths never share a headline.

### Block 1 — Ablation (pilot)
pony-only vs BT-only θ (no α-anchor) vs α·pony+β (full PaRC); full O(K²) duels vs adaptive O(K log K) merge-sort/dueling-bandit (target ≤2% NDCG loss at ≤15% of comparisons); symmetrized vs one-directional s_ij (position-bias cost); λ-sweep curve. Each vs pony, paired bootstrap.

### Block 2 — Comparison (8 domains, 10k users; GPU)
PaRC vs **pony pointwise posterior + 8 official baselines + RankGPT-style pairwise rerank + short-context listwise rerank** (≥11 comparison methods). The RankGPT/listwise LLM rerankers are **compute-matched** to PaRC (same per-user prompt/comparison budget) AND also placed on the compute-normalized axis (Block 4), so PaRC cannot win merely by spending more comparisons. Paired Holm-bootstrap; target: PaRC **non-inferior to pony** everywhere (≥ pony − ε) AND beats the strongest baseline in ≥6/8 domains with the comparative lift concentrated where β is non-trivial. Losses reported honestly.

### Block 3 — Mechanism
β concentrates on near-tie candidate pairs; NDCG gains localize at the top-k boundary; decompose lift into "better-calibrated BT utility" vs "non-BT cyclic structure"; per-event correlation of intransitivity with lift.

### Block 4 — Compute-normalized evaluation (defuses "expensive calibration layer")
Report NDCG@10 lift **per 1k prompts** and **per GPU-hour** for PaRC vs pony vs (compute-matched) RankGPT-pairwise vs short-listwise. **The METHOD headline requires the M0 validation lift to carry to test (non-inferiority + significant lift where present)**; the compute-normalized comparison + the intransitivity finding *strengthen* that headline (they are not independent escape hatches for a method that failed M0). If the method path is killed at M0, these same measurements become the **diagnostic/negative-result** contribution under a separate, clearly-labeled framing (per Block 0's path separation).

### Block 5 — Robustness / reproducibility
Position-bias controls (symmetrized + neutral labels + title-truncation + repeated-duel CIs on s_ij). **What is seeded (≥20 seeds for paper-result rows):** the LLM forward pass is *deterministic* (vLLM greedy, temp→0), so we do NOT average over model randomness; the meaningful stochastic sources we seed are (a) the adaptive **duel subsampling / merge-sort scheduling**, (b) the **λ/α validation-selection bootstrap**, and (c) **prompt-order/label perturbations** (A/B position, label tokens). Report mean ± CI over these. Optional backbone transfer reusing pony's Mistral/Llama scores as anchors.

## Baselines (≥8 satisfied)
8 official (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLM-ESR, ProEx, ProMax, RLMRec) + pony pointwise + RankGPT-pairwise + short-listwise = 11.

## Milestones + decision gates
- **M0** (toys kill-gate): proceed/kill. ~6–12 GPU-h (short prompts, adaptive duels, 10k users + phenomenon sub-sample).
- **M1** (8-domain comparison): ≥6/8 beat strongest baseline + PaRC **non-inferior to pony everywhere** (≥ pony − ε, ε = 0.002 abs NDCG@10, the same pre-set margin as Block 2 — we do NOT assert guaranteed test dominance), with the comparative lift concentrated where β is non-trivial.
- **M2** (ablation + mechanism + compute-normalized + robustness, ≥20 seeds): paper-ready evidence, all evidence-labeled.
- **M3** (paper-write → auto-review-loop ≥8 → citation-audit → paper-claim-audit).

## Compute & timeline (1×RTX4090, GPU-queued behind pony's 3-backbone run, then truce)
Pairwise duels are short + vLLM-batched ⇒ far fewer total tokens than CC-PACE's 32k listwise panels. M0 ~6–12 GPU-h; full 8-domain ~3–5 GPU-days. CPU/design (this plan, scorer code, BT MLE, analysis) proceeds now in parallel.

## Evidence discipline
Labels smoke→pilot→diagnostic→controlled→official→paper-result; only paper-result rows enter the paper; significance required for every claim; configs+seeds committed; large artifacts server-side with manifests; light evidence to git.

## Pre-registered KILL (repeat)
If toys M0 shows validation λ→~0 OR no significant NDCG lift over pony → **KILL PaRC, do not scale to 8 domains**; document as a second characterized negative result and escalate to a fresh ARIS research-refine round.
