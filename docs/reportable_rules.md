# Reportable Rules

- Smoke outputs are never reportable.
- Pilot outputs are marked `NON_REPORTABLE` and `pilot_reportable=false`.
- Mock methods are never reportable.
- Stub methods are never reportable.
- Diagnostic-only artifacts are never reportable.
- Sampled pilot data cannot be used by reportable configs unless a future protocol explicitly
  allows a labeled pilot-reportable study.
- API calls must be disabled by default.
- LoRA/QLoRA training must be disabled for pilot configs.
- Paper tables must be generated from metrics files, not manually typed.
