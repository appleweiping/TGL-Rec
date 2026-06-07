# Baselines

The paper-facing comparison uses **eight official LLM4Rec baselines** evaluated under a
single shared same-candidate protocol across **eight domains**. The frozen evidence
(per-(domain,baseline) metrics, provenance, coverage audits) and the 64-row master table
live in [`../data/pony_official_baselines/`](../data/pony_official_baselines/); see also
`configs/baselines/pony_official_external.yaml`, `docs/reference_baseline_fidelity.md`,
and `docs/reference_method_adaptation_map.md`.

The eight official baselines (all official Qwen3-8B, completed across the 8 domains):

- `llm2rec`;
- `llmesr`;
- `llmemb`;
- `rlmrec`;
- `irllrec`;
- `elmrec`;
- `proex`;
- `promax`.

Not part of the shared protocol:

- `setrec` — only a single-domain run exists, so it is excluded from the main comparison;
  the 8th slot is held by `llmemb`.

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
