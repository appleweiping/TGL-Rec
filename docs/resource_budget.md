# Resource Budget

Phase 7 resource estimates are written before pilot execution.

The pilot budget forbids:

- real API calls;
- LoRA/QLoRA training;
- full MovieLens/Amazon runs;
- paper-scale seed sweeps.

The estimator writes candidate-score counts, method counts, memory estimates, and NON_REPORTABLE
status to `resource_estimate.json`.
