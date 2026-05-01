# Baselines

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
