# Paper Experiment Matrix

This matrix is planned for `protocol_v1` but not executed in Phase 8.

## Accuracy

Popularity, BM25, MF/BPR, SASRec, TemporalGraphEncoder, TimeGraphEvidenceRec without dynamic
encoder, and TimeGraphEvidenceRec with dynamic encoder. Random remains useful for pilot sanity
checks but is not in the Phase 8 paper accuracy queue.

## Ablations

Full, without temporal graph, without transition edges, without time-window edges, without time-gap
tags, without semantic similarity, without dynamic encoder, graph-only, and text-only.

## Datasets

MovieLens full and Amazon multi-domain full are paper-scale later. Sampled variants are pilot only.

## Phase 8 Launch Artifacts

The paper matrix is represented by configs under `configs/experiments/paper_*.yaml`, the launch
manifest at `outputs/launch/paper_v1/launch_manifest.json`, and planned jobs at
`outputs/launch/paper_v1/jobs.jsonl`.

No Phase 7 pilot number or Phase 8 launch artifact is a paper result.
