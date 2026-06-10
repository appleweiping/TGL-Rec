# CLAUDE.md — TGL-Rec

You are working on TGL-Rec: Temporal Graph-to-Language Retrieval for Need-Aware Sequential Recommendation.

## Mandatory Read Order
1. `AGENTS.md` — authoritative engineering contract (620 lines)
2. `README.md` — full project documentation
3. `docs/technical_design.md` — method math and architecture
4. `docs/EXPERIMENT_PLAN.md` — ARIS experiment plan (6 blocks)
5. `docs/phase10_master_plan.md` — current phase plan
6. `docs/codex_project_memory.md` — durable memory
7. This file

## Quick Context
- GitHub: https://github.com/appleweiping/TGL-Rec
- Stage: Phase 10 — reportable scoring path implemented, awaiting server deployment
- Branch: `codex/phase9e-lora-rerank-eval` (active), `main` (stable)
- Core code: `src/llm4rec/` (active framework), `src/tglrec/` (legacy CPU tools)
- Configs: `configs/` (790+ YAML files)
- Tests: `tests/unit/` + `tests/smoke/` + `tests/test_reportable_modules.py`

## Critical Rules
1. **Every step/stage/fix/contribution → update agentmemory + project docs + README, and commit/push.** No exceptions; do it as you go, not batched at the end.
2. Never fabricate experiment results or claim unverified improvements. The old paper "SOTA" claim was unsupported — do not repeat it.
3. All baselines AND our method run under the SAME candidate protocol (frozen): same-candidate, 101 candidates/user, Qwen3-8B backbone, metrics HR@5/10/20 + NDCG@5/10/20 + MRR.
4. Evidence levels: smoke → pilot → diagnostic → controlled → official → paper-result.
5. **8 official baselines × 8 domains, frozen & complete** (see `data/pony_official_baselines/`, 64-row master table): llm2rec, llmesr, llmemb, rlmrec, irllrec, elmrec, proex, promax. (8th slot = `llmemb`, NOT setrec.)
6. No paper claims without statistical significance testing (paired tests, multi-seed).
7. TGL-Rec is INDEPENDENT from Pony — never mix methods or modify Pony files. Pony's C-CRP rows are excluded from our baseline evidence.
8. **Local ↔ server alignment + git discipline:** experiments run on the server; commit/push only from LOCAL; package lightweight evidence (metrics/provenance/summaries, not multi-GB scores/ckpts) back to local after each server run. Don't stop unless server disk full / API failure / unfixable review reject / user says pause.
9. **Three-way discussion protocol:** when redesigning method or stuck, run Opus-lead + Opus#2 + GPT-5.5 xhigh under ARIS (independent proposals → adversarial critique → synthesis → ≥8/10 design gate). See global memory `multi-agent-discussion-rule`.

## Research Question
Do LLM rerankers score candidates by user-conditional relevance, or do they collapse to popularity/semantic-similarity? The old temporal-graph premise ("LLMs ignore order") was never validated and is abandoned.

## Method Architecture — CC-PACE (current; supersedes RW-PMI and the old TDIG core)
Decided 2026-06-07 via a ≥20-round tri-agent escalation (Opus-lead + Opus-#2 + GPT-5.5 xhigh);
full design `docs/method_v2_decision_CC-PACE.md`, raw discussion in
`outputs/method_redesign_v2_discussion/`. RW-PMI (`docs/redesign_decision_RW-PMI.md`) and the earlier
TDIG/need-gate design are both superseded.

**CC-PACE = Collaborative-Conditioned Panel-Anomaly Calibrated Exchangeability reranker.** One
mechanism: a single Qwen3-8B forced-choice listwise judge over the 101-panel (unified schema +
randomized label IDs + long-term profile slots + rendered frozen-CF neighbor evidence tokens) emits
per-candidate evidence E_judge; rank by the residualized statistic
`T_u(c) = E_judge(c) − m̂_LOO(content, facet, log-pop, CF_emb, r_CF, CF_cluster)` (symmetric LOO
isotonic). CF is the conditioning σ-field (frozen, NOT a score head) — ablating CF tokens = text-only
PACE with zero code-path change, which is the non-stitch proof. Dual null max(p_pop, p_sem) +
split-conformal as a calibration/abstention layer (honest: conformal does NOT change within-panel
NDCG; residualization + the listwise Plackett-Luce-trained LoRA drive ranking). Honest top risk:
promax (beauty SOTA 0.1506) — beauty is profile-expressible, so profile slots are mandatory.

**IMPLEMENTED** in `src/llm4rec/methods/cc_pace/` (config/schema/judge/hf_judge/residualizer/
conformal/cf_conditioning/statistic), ranker `llm4rec.rankers.CCPaceRanker`, trainer
`llm4rec.trainers.cc_pace_trainer`, driver `scripts/cc_pace_beauty.py` + `scripts/run_cc_pace_beauty.sh`
(GPU-gated), tests `tests/unit/test_cc_pace_*.py` (10, CPU-passing). **To run experiments + write the
paper, follow `docs/HOW_TO_RUN_CC_PACE.md`** — only CF-provider artifacts + profile-slot data-prep
remain to wire on the server (documented seams; `--mock` proves the full pipeline).

## Current Phase (Phase 10) — CC-PACE rollout
- **8 domains**: sports, toys, home, tools (10k users) + books, electronics, movies (10k) + beauty (973).
- **Plan**: run CC-PACE on **beauty first** (smallest, fast). Beauty SOTA bar = promax NDCG@10=0.1506.
  - If beauty reaches SOTA → roll the SAME method out to the other 7 domains.
  - If NOT → re-run the three-way ARIS discussion (Opus+Opus+GPT-5.5), redesign, and re-run beauty (formal) until it is genuinely SOTA on beauty before scaling.
- **How to run / handoff: `docs/HOW_TO_RUN_CC_PACE.md`.** Driver `scripts/cc_pace_beauty.py`; server
  runner `scripts/run_cc_pace_beauty.sh` (self-gates on ≥17GB free GPU, never preempts).
- Go/kill thresholds + full design: `docs/method_v2_decision_CC-PACE.md`.
- (Superseded, kept for history: `scripts/rwpmi_zeroshot_beauty.py`, `docs/redesign_decision_RW-PMI.md`.)

## After the performance table — required paper experiments
Once the main performance result is done (CC-PACE vs 8 baselines × 8 domains), THREE experiments are
required before submission (advisor-specified) — see `docs/paper_followup_experiments.md`:
1. **Observation** (motivation): show the popularity-collapse / mis-calibration phenomenon. Use the
   **baseline models** (no paid/SOTA general model needed), ~2 domains is enough (ICLR precedent),
   reuse earlier observation material if found. Output a clean figure/table.
2. **Ablation**: toggle off each RW-PMI component; a component whose removal doesn't hurt (or helps)
   is badly designed — report honestly.
3. **Hyperparameter analysis**: sweep lr / α / δ / τ / LoRA rank etc., plot performance curves
   (matplotlib) to show stability.
Plus a **framework overview figure** (PPT or LLM-generated). Main table + these 3 + overview ≈ ready
to write and submit; other experiments supplemented later. **Do NOT start these until the
performance table is complete.**

## Server Access

**Local is primary, server is experiment-only.** All code/docs/commits happen locally. Server only does `git pull` → run → output results. Never commit from server.

Remote GPU server `pony-rec-gpu` is now directly accessible via SSH (key-based auth configured):
- **SSH command**: `ssh pony-rec-gpu`
- **Host**: `125.71.97.70`, Port `15302`, User `ajifang`
- **GPU**: NVIDIA RTX 4090 (49GB VRAM)
- **TGL-Rec server path**: `~/projects/TGL-Rec/` (independent from Pony)
- **TGL-Rec conda env**: `tglrec` (independent from Pony's `qwen_vllm`)
- **Pony path (READ-ONLY)**: `~/projects/pony-rec-rescue-shadow-v6/`
- **Local project path**: `D:\Research\TGL-Rec`
- **Deploy script**: `scripts/deploy_server.sh`

## Agent Roles
- **Codex**: Primary execution engine (server commands, parallel runs)
- **Claude/Opus**: Architecture review, paper writing, complex reasoning
- **OpenCode**: Implementation, testing, doc updates
