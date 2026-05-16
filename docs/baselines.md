# Baselines

Active paper-facing official baselines now come from the Pony/Uncertainty same-candidate suite, not from a new TGL-Rec-specific senior-reference queue. See `configs/baselines/pony_official_external.yaml`, `docs/reference_baseline_fidelity.md`, and `docs/reference_method_adaptation_map.md`.

Completed main-table candidates after manifest/provenance/score-gate checks:

- `llm2rec`;
- `llmesr`;
- `llmemb`;
- `rlmrec`;
- `irllrec`;
- `elmrec`;
- `proex`.

Pending/planned:

- `promax`, excluded from completed main tables until all declared domains pass.

Blocked/replaced:

- `setrec`, replaced by `elmrec`, `proex`, and `promax`.
Implemented smoke/pre-experiment baselines:

- random;
- popularity;
- BM25;
- BPR-MF smoke trainer/ranker;
- deterministic first-order Markov transition sequential baseline marked `reportable: false`.

Formal sequential baselines:

- SASRec interface exists but raises `NotImplementedError` until a validated implementation or
  wrapper is added.
- GRU4Rec interface exists but raises `NotImplementedError` until a validated implementation or
  wrapper is added.

Smoke baselines must not be reported as formal SASRec/GRU4Rec results.
