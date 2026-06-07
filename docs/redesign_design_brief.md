# TGL-Rec Method Redesign — Design Brief for Independent Proposal

You are an independent senior ML-systems researcher contributing to a top-conference
(KDD/WWW/SIGIR/RecSys) submission. Produce ONE original method proposal. Be adversarial
toward weak ideas, including the existing one. Do NOT propose a "stitch" of existing tricks —
aim for a genuinely novel, defensible core mechanism.

## Setting (fixed, do not change)
- Task: LLM-based sequential recommendation, **candidate reranking**, same-candidate protocol,
  **101 candidates/user** (1 positive + 100 negatives), backbone **Qwen3-8B**.
- Metrics: HR@5/10/20, NDCG@5/10/20, MRR. Primary: NDCG@10 / MRR.
- 8 domains: sports, toys, home, tools (10k users each) + books, electronics, movies (10k) +
  beauty (973, supplementary, smallest → chosen as the first pilot domain because results are
  fast to see).
- 8 official baselines already frozen and strong (LLM4Rec methods, official code, Qwen3-8B):
  elmrec, irllrec, llm2rec, llmemb, llmesr, proex, promax, rlmrec. On the new domains the
  strongest baselines reach e.g. NDCG@10 ~0.18 (sports llmemb), MRR ~0.15.
- Compute: a sister project (Pony-Rec) already showed a *pointwise calibrated-relevance* method
  (C-CRP) ranks #1 vs all 8 baselines on these new domains — so beating these baselines IS
  achievable; the bar is real but not impossible.

## The existing method (TGL-Rec) and WHY IT IS FAILING — your starting evidence
Core idea: build a **Temporal Directed Item Graph (TDIG)** from train-only consecutive user
transitions (time-decayed edges, transition prob / PMI / lift / direction asymmetry); compute a
5-dim user "need-state"; a **26-parameter logistic "need-gate"** outputs α per (user,candidate)
mixing temporal-evidence vs semantic-similarity; Stage-1 hand-feature score over 101 candidates;
Stage-2 LoRA Qwen3-8B reranks top-K with the graph evidence translated to natural language.

Measured result (the only runs that exist, on a mixed debug set + movielens, few-hundred samples):
- temporal-evidence rerank NDCG@5 ≈ 0.065, transition-evidence ≈ 0.067
- **naive popularity NDCG@5 = 0.104, history-only rerank = 0.069 — i.e. the temporal evidence
  LOSES to popularity and to plain history.**
- The paper claims SOTA but the results tables are empty placeholders.

Diagnosed likely failure causes (challenge or confirm these):
1. **Signal sparsity**: Amazon inter-event gaps are weeks/months; a 7-day transition window +
   per-day decay leaves most candidates with zero transition edges → evidence vector ≈ 0 → the
   gate degenerates and the model falls back below popularity.
2. **Information bottleneck**: a 26-param linear gate over 10 hand-features throws away most signal;
   translating a handful of scalars into prose gives the LLM no real increment.
3. **Unvalidated premise**: "LLMs ignore temporal order" (the whole motivation) was never tested
   here; the pain point may be weak or absent.

## What we need from you (structure your answer EXACTLY like this)
1. **Verdict on TGL-Rec**: fixable-as-is / fixable-with-major-surgery / abandon-core-idea. One paragraph, decisive.
2. **Proposed method (the core)**: the ONE central mechanism, stated as a single falsifiable claim,
   then the math/architecture. Make explicit what is genuinely new and why it is not a stitch.
3. **Why it beats these 8 baselines specifically** (and why it should win on beauty first): tie to
   what the baselines structurally cannot do.
4. **Why it works under signal sparsity** (the killer of the old design): be concrete.
5. **Minimal falsification experiment on beauty**: the smallest run that would tell us in <1 day
   whether the core mechanism has life, and the exact go/kill metric threshold.
6. **Top 3 risks** and how each is mitigated or detected early.
7. **Novelty/defensibility**: the one-sentence rebuttal to a reviewer who says "this is just X+Y".

Constraints: must run on Qwen3-8B + 101-candidate rerank protocol; must be implementable on a
single GPU server; LoRA is allowed; external heavy pretraining is NOT. Prefer mechanisms that are
auditable/interpretable (a paper plus). Be concrete and quantitative. ~800-1200 words.
