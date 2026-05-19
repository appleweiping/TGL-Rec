# CLAUDE.md — TGL-Rec

You are working on TGL-Rec: Temporal Graph-to-Language Retrieval for Need-Aware Sequential Recommendation.

## Mandatory Read Order
1. `AGENTS.md` — authoritative engineering contract (620 lines)
2. `README.md` — full project documentation
3. `PROJECT_CHARTER.md` — research thesis and RQs
4. `EXPERIMENTS.md` — frozen experiment protocol
5. `docs/phase10_master_plan.md` — current phase plan
6. This file

## Quick Context
- GitHub: https://github.com/appleweiping/TGL-Rec
- Stage: Phase 10 — four-domain same-candidate protocol experiments
- Branch: `codex/phase9e-lora-rerank-eval` (active), `main` (stable)
- Core code: `src/llm4rec/` (active framework), `src/tglrec/` (legacy CPU tools)
- Configs: `configs/` (790+ YAML files)
- Tests: `tests/unit/` + `tests/smoke/`

## Critical Rules
1. Never fabricate experiment results or claim unverified improvements
2. All baselines must run under the SAME candidate protocol (frozen)
3. Evidence levels: smoke → pilot → diagnostic → controlled → official → paper-result
4. 7 official baselines completed: llm2rec, llmesr, llmemb, rlmrec, irllrec, elmrec, proex
5. No paper claims without statistical significance testing
6. Follow stage gates in `configs/stage_gates.yaml`

## Research Question
Does the LLM actually use temporal/sequential signals, or just semantic similarity?
Proposed answer: TDIG (Temporal Directed Item Graph) + graph-to-language evidence + gated reranker.

## Current Phase (Phase 10)
- Four-domain experiments: Beauty, Books, Electronics, Movies
- Same-candidate protocol frozen
- LoRA/QLoRA fine-tuning of Qwen3-8B implemented
- Pony official baselines integrated
- Remaining: run final experiments, fill paper tables, write results section

## Agent Roles
- **Codex**: Primary execution engine (server commands, parallel runs)
- **Claude/Opus**: Architecture review, paper writing, complex reasoning
- **OpenCode**: Implementation, testing, doc updates
