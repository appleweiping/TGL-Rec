# Frozen Experiment Protocol

Phase 7 freezes the pre-experiment protocol. This document defines the comparison contract before
paper-scale runs are launched.

## Datasets

- `tiny`: smoke only.
- MovieLens-style sampled: pilot only, `pilot_reportable=false`.
- MovieLens full: paper-scale later.
- Amazon multi-domain sampled: pilot later.
- Amazon multi-domain full: paper-scale later.

## Splits

The default split is leave-one-out. Temporal split remains optional and must be declared in config.
Timestamp ties are resolved by timestamp and item id for deterministic ordering. User history must
exclude the target item and any event at or after the prediction timestamp. Split artifacts are
saved and reused across methods.

## Candidate Protocol

Full ranking is used only when feasible. Pilots use sampled fixed candidates with the same candidate
artifact for every method. The target item must be included, sampled negatives are seed-fixed, and
candidate artifacts are saved.

## Metrics

Required metrics include Recall@K, HitRate@K, NDCG@K, MRR@K, coverage, novelty, diversity,
long-tail ratio, validity/hallucination for LLM outputs, efficiency metrics, and segment metrics.

## Methods

The frozen method list is Random, Popularity, BM25, MF/BPR, SASRec, TemporalGraphEncoder,
TimeGraphEvidenceRec, ablations, and disabled LLM diagnostics unless explicitly allowed later.

## Seeds

Pilot seeds are `[0]`. Paper-scale seeds later are `[0, 1, 2, 3, 4]`. Seeds must be saved in configs
and metrics.

## Reportability

Smoke runs are `reportable=false`. Pilot runs are `pilot_reportable=false`. Paper-scale configs may
be `reportable=true` only after protocol freeze and readiness checks. Mock methods, stub methods,
and diagnostic-only artifacts are never reportable.

## Leakage Rules

Evidence and encoders use train-only inputs for pilot/reportable configs. Temporal graph encoders
must not process future events. Prompts/evidence cannot contain target labels. Graph statistics must
not come from test interactions. Candidate construction may include the target only through the
declared protocol.
