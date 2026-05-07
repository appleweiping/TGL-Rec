# Phase Plan

Current phase: **Phase 10, observation-to-framework-to-system hardening**.

Older Phase 4-8 documents remain as historical scaffolding. Phase 9E made local
Qwen3-8B LoRA training/evaluation and official-baseline provenance concrete.
Phase 10 is the project-level consolidation needed before server-scale runs and
paper claims.

## Active Milestones

1. **M0 Observation Reproduction**
   - Rerun fixed-label-mask control LoRA diagnostics.
   - Measure sequence/time sensitivity before claiming the observation.
2. **M1 Our Framework**
   - Build time-aware graph evidence as reusable modules.
   - Keep graph/evidence artifacts train-only and auditable.
3. **M2 Complete Recommender System**
   - Compare against classical, sequential, graph, text, LLM, and official
     reference baselines.
   - Keep official baseline algorithms faithful; unify protocol only.
4. **M3 Four Large Domains**
   - Import `beauty`, `books`, `electronics`, and `movies` same-candidate tasks.
   - Preserve 10k-user/100-negative event alignment.
5. **M4 Top-Conference Gate**
   - Run literature and reviewer audits.
   - Block reportable tables until baselines, seeds, leakage checks, and
     statistical tests are complete.

The detailed plan is in `docs/phase10_master_plan.md`.

## Current Non-Reportable Boundary

The current `time_graph_evidence` and `time_graph_evidence_dynamic` paper-config
entries are infrastructure placeholders until the method card is upgraded and
locked metrics exist. `scripts/validate_experiment.py` intentionally blocks
paper configs that include these methods as reportable methods.

## Server Entry Point

Generate the server execution plan with:

```bash
python scripts/plan_four_domain_runs.py \
  --external-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output outputs/plans/four_domain_server_plan.json \
  --shell-output outputs/plans/four_domain_server_runbook.sh
```

Then inspect the generated files before executing long jobs.
