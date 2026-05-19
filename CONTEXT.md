# Project Context — TGL-Rec

## Current State (as of 2026-05-19)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/TGL-Rec |
| Commits | 51 |
| Branch | codex/phase9e-lora-rerank-eval (active) |
| Stage | Phase 10 — four-domain experiments |
| Official baselines | 7 completed (same-candidate protocol) |
| Datasets | 4 domains (Beauty, Books, Electronics, Movies) |
| Method | TDIG + graph-to-language evidence + gated reranker |
| LLM | Qwen3-8B (LoRA/QLoRA fine-tuning) |
| Python | >=3.10 |
| License | MIT |

## Research Questions
1. Do LLM-based recommenders actually use temporal/sequential signals?
2. Can explicit temporal graph evidence improve LLM recommendation?
3. Does need-aware gating outperform uniform evidence injection?
4. How does the approach generalize across domains?

## Phase History
- Phase 1-4: Infrastructure, data preprocessing, baseline implementation
- Phase 5-7: TDIG construction, evidence generation, method implementation
- Phase 8: Diagnostics and ablation design
- Phase 9: LoRA fine-tuning, evaluation framework
- Phase 10 (CURRENT): Four-domain same-candidate protocol experiments

## Key Decisions
- Same-candidate protocol: all methods rank the same candidate set (no cherry-picking)
- Pony official baselines reused (shared infrastructure with TRUCE-Rec)
- Evidence levels enforced: no paper claims without controlled experiments
- Stage gates prevent premature advancement

## What's Next
- [ ] Complete four-domain experiments under frozen protocol
- [ ] Statistical significance testing (paired t-test, bootstrap CI)
- [ ] Fill paper tables with official results
- [ ] Write results and analysis sections
- [ ] Internal review gate before submission
