# TGL-Rec Paper Follow-up Experiments (post-performance roadmap)

**Status:** roadmap only — DO NOT start these until the main performance table is done.
**When this kicks in:** after the main result is complete = **our RW-PMI method vs the 8 official
baselines across all 8 domains** (the performance table; beauty first, then the other 7).
**Source:** advisor (学长) guidance, recorded 2026-06-07. With these three experiments + a framework
overview figure, the paper is essentially ready to write and submit; any remaining experiments are
supplemented afterward.

Three experiments are required before submission:

---

## 1. Observation (motivation experiment)

**Goal:** justify *why* this framework was conceived — the pain point / phenomenon that motivates the
design. (Advisor's words: "你为什么想到用 uncertainty 去做这个 framework".) For our RW-PMI method the
motivating observation is the **popularity-collapse / mis-calibration** story: pointwise LLM-reranker
scores track item popularity / verbosity rather than user-conditional relevance, so the
set-normalized marginal is the missing debiasing term. Show this empirically.

**Key methodology constraints (advisor, important):**
- **No paid / most-SOTA general model needed.** Observe the phenomenon **directly with the baseline
  models** (the LLM4Rec baselines and/or base Qwen3-8B). Data-volume requirement for observation is
  **low**.
- **Precedent:** an ICLR paper ran its observation using just **two baseline models** — that is
  enough. We do NOT need to observe with the strongest general model.
- **Don't need all domains.** Pick ~2 representative domains (the ICLR example used 2).
- **Reuse prior work if possible:** a version of this was already done earlier — "看别人的方法加上
  base 的观察,不要只看 base" (observe baselines + base, not base alone), possibly only on ~4 domains /
  8 baselines, a while ago. Find that material and reuse/extend rather than redoing from scratch.
- **Output:** a clean **figure or table** that makes the motivation obvious (e.g. score-vs-popularity
  correlation per baseline; how much a set-normalized marginal moves ranking; calibration gap).

## 2. Ablation study

**Goal:** remove each designed component and measure the effect. **If removing a component keeps
performance the same — or improves it — that component is badly designed** and must be reconsidered
(honest signal, report truthfully; do not hide a dead component).

**RW-PMI components to ablate (each toggled off vs the full method):**
- set-normalized marginal debiasing — off → raw conditional `log p(c|H)` only.
- residualized intent-witness gain — off → `α = 0`.
- token-length guard — off → `δ = 0`.
- training objective — InfoNCE over the 101-set vs. plain next-token LM vs. zero-shot (no training).
- negative sampling — **popularity-matched** vs. random/easy negatives.
- witness residualization — residualized-vs-PMI vs. raw witness gain.

Each ablation on the chosen domain(s), multi-seed, with significance testing. Map cleanly to the
method's named components so a reviewer sees each is necessary.

## 3. Hyperparameter analysis

**Goal:** sweep each hyperparameter and **plot a performance curve** to show the method is **stable**
(not a knife-edge that only works at one lucky setting).

**Example (advisor):** learning rate is one value, say `1e-3` → run it also at `1e-1, 1e-2, 1e-3,
1e-4, 1e-5` and plot NDCG@10 / MRR vs learning rate as a line chart.

**RW-PMI hyperparameters to sweep:** learning rate, `α` (witness-gain weight), `δ` (length penalty),
`τ` (InfoNCE temperature), LoRA rank, max history length, number of witnesses `M`. One line plot per
hyperparameter (metric vs value); a flat-ish curve = stable.

---

## Figures / plotting

- **Line charts and similar (ablation/hyperparameter):** `matplotlib`. Let an LLM write the plotting
  code — it's simple.
- **Framework overview figure:** draw by hand in PPT, or have an LLM generate it.

## Submission note

Main performance table + these three experiments + the overview figure ≈ ready to write the draft and
submit. Other experiments can be added later. See `docs/redesign_decision_RW-PMI.md` for the method
and `data/pony_official_baselines/` for the baseline reference table.
