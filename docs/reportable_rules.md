# Reportable Rules

Current Phase 10 reportability memory lives in
`docs/codex_project_memory.md`, `docs/phase10_master_plan.md`, and
`docs/reference_baseline_fidelity.md`. The Phase 8 notes below are historical
launch constraints and must not be read as a global ban on the later Phase 9E/10
LoRA work.

Reportability rules for launch:

- smoke runs are `reportable=false`;
- pilot runs are `pilot_reportable=false`;
- Phase 8 launch artifacts are plans, not results;
- paper configs may be `reportable=true` only when they are safe to launch later;
- mock, stub, skeleton, and Markov smoke methods are never reportable;
- diagnostic-only artifacts are never reportable;
- real API calls and LoRA training are disabled in Phase 8 paper configs;
- no paper table can contain manually typed metric values.

Pilot outputs and Phase 8 launch artifacts must not be cited as paper results.
