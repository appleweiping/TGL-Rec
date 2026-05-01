# Experiment Protocol

This repository is still in pre-experiment readiness mode. Reportable runs must use one resolved
config per run, one shared split protocol, one shared candidate protocol, one shared prediction
schema, and the shared evaluator.

Required before any reportable run:

- save `resolved_config.yaml`;
- save `environment.json`;
- save predictions as `predictions.jsonl`;
- export metrics as `metrics.json` and `metrics.csv`;
- record seeds, split strategy, candidate strategy, dataset, method, and run mode;
- generate paper tables only from saved metrics artifacts.

Smoke and diagnostic runs are marked `reportable: false`.
