# Leakage Protocol for Time-Aware Evidence

Phase 5 introduces a leakage validator for evidence-based LLM4Rec methods. The validator is intentionally conservative because temporal graph evidence can accidentally expose future interactions.

## Required Checks

- Evidence provenance must be present.
- Reportable evidence must be constructed from `train_only` artifacts.
- Diagnostic-only artifacts are blocked in reportable configs.
- Time-window evidence cannot include events after the prediction timestamp.
- User history cannot contain the target item.
- LLM prompts cannot include the target label.
- Mock providers and stub encoders are blocked in reportable configs.
- Smoke, skeleton, and Markov-style methods must remain `reportable=false`.

## Current Scope

Phase 5 uses tiny smoke data and deterministic evidence scoring. The validator is already wired into the ranker so future real components inherit the same checks before paper-scale runs.
