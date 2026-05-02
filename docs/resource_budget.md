# Resource Budget

Resource estimates are planning artifacts, not guarantees.

Phase 7 writes NON_REPORTABLE pilot estimates before pilot execution. Phase 8 writes paper launch
estimates before any paper-scale job is allowed to run:

```bash
python scripts/estimate_paper_resources.py --manifest outputs/launch/paper_v1/launch_manifest.json
```

The Phase 8 budget estimates:

- number of planned jobs;
- trainable and eval-only jobs;
- CPU and GPU hours;
- disk usage;
- prediction, checkpoint, and table storage;
- API calls, which must be `0`;
- LoRA training jobs, which must be `0`.

Every Phase 8 budget file records `NO_EXPERIMENTS_EXECUTED_IN_PHASE_8 = true`.
