# Failure Modes

Paper-scale launch preparation separates code readiness from data readiness.

Tracked failure categories:

- missing full datasets;
- partial dataset schema or timestamp quality;
- missing protocol manifests;
- paper config safety violations;
- API-enabled or LoRA-enabled jobs in paper configs;
- job queue rows not marked `planned`;
- planned output directories already containing result artifacts;
- incomplete runs requested for result locking.

Phase 8 failure reports are launch blockers only. They are not experimental results.
