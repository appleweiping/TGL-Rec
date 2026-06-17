# Method — PaRC (Pairwise-Relational Calibration)

> **Status / provenance.** Method description + pre-registered protocol for the current TGL-Rec headline, PaRC, which **replaces the failed CC-PACE** (beauty zero-shot NDCG@10 0.1108, LoRA ≈0.086, both < ProEx bar 0.1506, and slow: 32k-token listwise HF judge ≈16h/973, vLLM-non-viable). PaRC cleared the ARIS research-refine gate (Codex GPT-5.5 xhigh v2: NOVELTY 8 / FEASIBILITY 7) and the experiment-plan gate (Codex 8/8/7/7/8); core code is implemented + CPU-tested (`src/llm4rec/methods/parc/`, 43 tests) and the vLLM duel adapter + M0 driver are built. **The method is NOT yet empirically validated** — the M0 kill-gate (below) is GPU-queued behind pony's run and has not run. This section therefore describes the method and the *pre-registered* protocol; it makes NO claim that PaRC beats any baseline. Frozen protocol: 8 Amazon domains, 10k users (beauty 973), 101 same-candidates/event, Qwen3-8B, HR/NDCG@{5,10,20}+MRR, paired Holm-bootstrap. Design refs: `refine-logs/RESEARCH_REFINE_PaRC_2026-06-17.md`, `refine-logs/EXPERIMENT_PLAN_PaRC.md`.

## 1. The open lane — comparative structure a pointwise posterior cannot see
The sibling project (pony) owns the **pointwise** posterior: an absolute relevance estimate `pony_i = P(relevant | history, candidate_i)` ranked raw. By construction a pointwise scorer cannot express **between-candidate** structure — preferences that exist only when two candidates are compared head-to-head. CC-PACE tried to capture comparison via a 101-candidate listwise panel, which failed (a 32k-token attention/credit-assignment sink with no clean per-candidate gradient, and batch≈1 defeating vLLM). PaRC targets the same comparative signal through a **cheap, vLLM-batchable, theory-grounded estimator** that is *anchored on* (never re-derives) the pony posterior.

## 2. Symmetrized pairwise duels
For a user with history `H` and two candidates `i, j`, a short (~300–600 token) prompt asks "given `H`, is item A or B the more likely next interaction?". To cancel position bias, the same pair is presented in both orders and the logits symmetrized:
```
   s_ij = ½ ( logit(A=i, B=j) − logit(A=j, B=i) ),
```
where `logit(A,B)` is the model's log-odds that the slot-A item is the more likely next interaction. The discarded antisymmetric part `b_ij = ½(logit(A=i)+logit(A=j))` is a **free position-bias diagnostic** (zero if unbiased). Duels are short and batch under vLLM continuous batching — the structural fix for CC-PACE's batch≈1 failure.

## 3. Anchored low-rank Bradley–Terry field
The symmetrized comparisons are modeled as a Bradley–Terry field `s_ij ≈ θ_i − θ_j + ε_ij`, with the latent strength decomposed against the pony anchor:
```
   θ_i = α · pony_i + β_i.
```
`β_i` — the **comparative residual** — is the headline object: the correction the frozen LLM makes *only* when forced to compare, i.e. the part of the order not captured by **this** pointwise posterior under the **same** frozen LLM + prompt family. (We state this precisely; we do NOT claim β is absent from *any* conceivable pointwise scorer.) `θ` is fit by symmetrized BT least-squares (Thurstone-style), reporting `Var(β)` and the order-variance fraction explained by β as diagnostics.

## 4. Validation-gated mixture (the real floor)
The final score is an explicit, validation-selected mixture:
```
   score_i = pony_i + λ · β_i,   λ ≥ 0 selected on validation, λ = 0 allowed.
```
Because `λ = 0` recovers the pony order exactly, **the empirical floor is pony's score** (a true floor, not an assumption): if the comparative residual does not help, validation selects `λ → 0` and PaRC reduces to pony. The claim is the right inequality — *anchor + a validation-gated comparative correction ≥ anchor* — never an unconditional dominance assertion.

## 5. O(K log K) adaptive duel scheduler
A full pairwise field is `O(K²)` duels/user (`K=101` ⇒ ~5050). PaRC instead anchors the order at pony's posterior and runs an `O(K log K)` dueling-bandit / merge-sort schedule that concentrates comparisons near the **top-`k` boundary**, where NDCG@10 errors actually cost (~600–1300 short prompts/user — fewer total tokens than CC-PACE's listwise panels, fully vLLM-batched). The full `O(K²)` mode is retained for the phenomenon sub-study on a capped user sub-sample.

## 6. Pre-registered M0 kill-gate (PENDING — the make-or-break)
PaRC's central risk is the reviewer objection that a frozen LLM's "A or B" preference is just two absolute scores subtracted (BT-transitive ⇒ reproducible by *some* pointwise utility ⇒ "you reinvented pony at `O(K log K)`× cost"). This is settled **empirically, up front**, on a pilot domain (a *winning* domain with clear margin + top-k headroom — explicitly **NOT** beauty), with a **two-path** gate, all decisions on validation / a disjoint held-out pilot (official TEST untouched until the 8-domain comparison):
- **Method-success path → proceed:** PaRC beats pony by **≥ +0.005 NDCG@10 with significant paired bootstrap** AND validation `λ` does not collapse to ~0. On TEST the claim is *non-inferiority* vs pony (≥ pony − ε, ε = 0.002 abs) plus the lift where significant — no asserted test dominance.
- **Scientific-negative / diagnostic path:** if `λ → ~0` OR no significant lift, the method is **KILLED** (no 8-domain scale-up) — but the *characterized finding* about the LLM pairwise-preference field (β's orthogonality to the pointwise posterior, cyclic-intransitivity structure vs a strict BT + heteroskedastic-noise null, compute cost) is a separate, honestly-labeled diagnostic contribution. The two paths never share a headline.

The intransitivity finding (a significant non-BT cyclic residual ⇒ by the BT representation theorem no pointwise utility reproduces the order) is claimed **only** if the cyclic residual is significant AND predictive of the lift — separated explicitly from BT-transitive improvement and from β-residual-over-pony.

## 7. Compute-normalized evaluation
Because the objection "expensive calibration layer, not a new method" is fatal if unaddressed, every comparison reports NDCG@10 lift **per 1k prompts** and **per GPU-hour**, against pony, a RankGPT-style pairwise reranker, and a short-context listwise reranker, all compute-matched. The headline must show a strong lift, OR compute-normalized value, OR a scientific finding (the intransitivity result) that stands even at modest gains — established only after M0 proceeds.
