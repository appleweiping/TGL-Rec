# Pilot Failure Modes

Phase 7 captures failures instead of hiding them.

Failure categories:

- `skipped_dependency`: optional dependency such as PyTorch is unavailable.
- `validation_error`: config, schema, split, candidate, or leakage validation failed.
- `runtime_error`: method execution failed.

Every pilot run writes `failure_report.json`. A failure in pilot does not create a paper claim; it
creates a readiness task.
