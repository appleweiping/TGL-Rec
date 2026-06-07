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

## Method Architecture — RW-PMI (current; old TDIG core ABANDONED)
Decided 2026-06-07 via tri-agent discussion (`docs/redesign_decision_RW-PMI.md`). The TDIG +
26-param need-gate + graph-to-prose design is dropped (failed structurally: near-zero transition
edges on sparse Amazon data → lost to popularity). New method:
1. **set-PMI core**: `log p(c|H) − log p(c|null-user)` normalized **per-candidate inside the 101-set** (real within-list popularity debiasing).
2. **residualized intent-witness gain**: leakage-free witness from history only, residualized vs PMI (kept only if it adds ≥0.007 NDCG@10 and corr<0.80).
3. **token-length guard**; train with **InfoNCE over the 101-set + popularity-matched negatives** (LoRA), only after the zero-shot kill test passes.

## Current Phase (Phase 10) — RW-PMI rollout
- **8 domains**: sports, toys, home, tools (10k users) + books, electronics, movies (10k) + beauty (973).
- **Plan**: run RW-PMI on **beauty first** (smallest, fast). Beauty SOTA bar = proex NDCG@10=0.1506.
  - If beauty reaches SOTA → roll the SAME method out to the other 7 domains.
  - If NOT → re-run the three-way ARIS discussion (Opus+Opus+GPT-5.5), redesign, and re-run beauty (formal) until it is genuinely SOTA on beauty before scaling.
- Kill-test code: `scripts/rwpmi_zeroshot_beauty.py` + `scripts/run_rwpmi_beauty.sh` (self-gates on ≥17GB free GPU, never preempts).
- Go/kill thresholds: `docs/redesign_decision_RW-PMI.md`.

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
