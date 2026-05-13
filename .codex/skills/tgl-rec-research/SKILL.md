---
name: tgl-rec-research
description: Operate the TGL-Rec Phase 10 LLM4Rec research workflow. Use when working in the TGL-Rec repository on method implementation, baselines, four-domain same-candidate experiments, server handoffs, evaluation/reportability, reproducibility checks, paper-readiness gates, or durable project documentation under docs/.
---

# TGL-Rec Research

## Operating Contract

Treat TGL-Rec as a publishable research system. Do not create toy demos, fabricate results, weaken baselines, mix protocols, or write paper claims without saved experiment artifacts.

Start every nontrivial task by reading:

1. `docs/codex_project_memory.md`
2. `docs/phase10_master_plan.md`

For tasks touching results, baselines, claims, protocol fidelity, or method novelty, also read the relevant docs named in `docs/codex_project_memory.md`, especially the server runbook, four-domain protocol, baseline fidelity notes, method adaptation map, and time-graph evidence card.

Use `src/llm4rec/` as the active framework. Treat `src/tglrec/` as legacy or compatibility tooling unless the task explicitly targets it.

## Workflow

1. Classify the task: implementation, experiment plan, server handoff, baseline adaptation, evaluation/reportability, documentation, or reviewer gate.
2. Load the smallest reference below that matches the task.
3. Preserve the frozen evaluation protocol: same candidates, same event IDs, same evaluator, same prediction/schema contracts.
4. Implement through existing unified modules, YAML configs, and thin scripts. Avoid one-off experiment code unless the user explicitly asks for it.
5. For complex tasks, use available multi-agent review only when the current runtime policy permits it. If not permitted or blocked, do a self-review pass for leakage, fairness, reproducibility, and reportability.
6. Validate with targeted local commands and tests. For server work, give exact commands and wait for pasted evidence instead of inferring success.
7. Update durable docs when direction, commands, baseline status, server outcomes, or reportability gates change.
8. If local work is complete and the user has not asked otherwise, commit and push so the shared server can `git pull`.

## References

- Read `references/phase10-gates.md` for milestone gates, novelty pressure tests, and when experiments can be considered complete.
- Read `references/baseline-evaluation.md` when touching baselines, prediction files, imported scores, metrics, protocol artifacts, or paper tables.
- Read `references/server-handoff.md` when preparing commands for the shared GPU server or interpreting pasted server logs.
- Read `references/completion-checklist.md` before finalizing a complex task.

Prefer the live repository docs over this skill whenever they conflict. This skill is a routing and discipline layer, not a replacement for the active project memory.
