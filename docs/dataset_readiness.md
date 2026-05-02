# Dataset Readiness

Phase 8 checks full-dataset availability before any paper-scale experiment can run.

The readiness command is metadata-only:

```bash
python scripts/check_dataset_readiness.py --config configs/datasets/movielens_full.yaml
python scripts/check_dataset_readiness.py --config configs/datasets/amazon_multidomain_full.yaml
```

It writes JSON under `outputs/launch/paper_v1/dataset_readiness/` and always records
`NO_EXPERIMENTS_EXECUTED_IN_PHASE_8 = true`.

Statuses:

- `READY`: data files exist, required columns are present, timestamps and item text are usable.
- `PARTIAL`: files exist but schema or quality blockers remain.
- `MISSING`: expected files are absent. This is a clean launch blocker, not a crash.

The checker validates users, items, interactions, domains, timestamp range, sparsity, missing
timestamps, duplicate interactions, users with too few events, and item metadata text coverage.

No data is downloaded automatically. Provide paths in the dataset config and rerun readiness.
