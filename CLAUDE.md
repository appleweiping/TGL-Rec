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
1. Never fabricate experiment results or claim unverified improvements
2. All baselines must run under the SAME candidate protocol (frozen)
3. Evidence levels: smoke → pilot → diagnostic → controlled → official → paper-result
4. 8 official baselines completed (all 4 domains): llm2rec, llmesr, llmemb, rlmrec, irllrec, elmrec, proex, promax
5. No paper claims without statistical significance testing
6. TGL-Rec is INDEPENDENT from Pony — never mix methods or modify Pony files
7. Follow stage gates in `docs/EXPERIMENT_PLAN.md`

## Research Question
Does the LLM actually use temporal/sequential signals, or just semantic similarity?
Proposed answer: TDIG (Temporal Directed Item Graph) + learned need-gate + evidence-augmented LoRA reranker.

## Method Architecture (4 components)
1. **TDIG**: Temporal directed item-item transition graph (train-only)
2. **NeedStateEncoder**: User temporal state (drift, transition pressure, gap, entropy)
3. **LearnedNeedGate**: 26-param logistic gate (when to trust temporal evidence)
4. **Two-Stage Scoring**: Stage 1 evidence scoring → Stage 2 LoRA Qwen3-8B reranking

## Current Phase (Phase 10)
- Four-domain experiments: Beauty, Books, Electronics, Movies
- Same-candidate protocol frozen
- Reportable scoring path implemented (need_state.py, need_gate.py, reportable_scorer.py)
- 8 Pony official baselines integrated (all complete)
- Next: deploy to server → observation experiment → gate training → LoRA training

## Server Access

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
