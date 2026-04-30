# TGLRec: Temporal Graph-to-Language Retrieval for Need-Aware Sequential Recommendation

Working title: **Beyond Similarity: Time-Aware Graph Translation for LLM-based Sequential Recommendation**.

This repository is intended to become a top-tier recommendation-systems research project, not a toy demo. The central question is:

> Do LLM-based recommenders follow temporal need transitions, or do they mostly retrieve semantically similar items? Can temporal graph-to-language evidence make sequential recommendation more robust, explainable, and inductive?

## Core hypothesis

Many LLM4Rec pipelines appear sequential because the prompt is ordered, but the model may still rely mainly on item semantic similarity and popularity. We test this directly, then build a method that separates:

1. **semantic similarity**: items look alike or share attributes;
2. **temporal transition / next need**: item B is likely after item A under a time window, even when B is not similar to A;
3. **user state drift**: the same user changes intent over sessions, weeks, or months.

## Planned method in one line

Build a temporal directed item-transition graph from user histories, retrieve time-aware paths around a user and candidate items, translate those paths into compact natural-language evidence, and use a lightweight gated reranker plus optional dynamic GNN / LLM channel to rank candidates.

## Main deliverables

- Diagnostic benchmark: sequence sensitivity, time sensitivity, and similarity-vs-transition stress tests for LLM4Rec and classic recommenders.
- Temporal Directed Item Graph (TDIG) construction with time-bucketed edge statistics.
- Temporal graph-to-language retriever and evidence translator.
- Need-aware reranker with a learned gate between semantic similarity and temporal transition evidence.
- Optional continuous-time dynamic GNN channel for inductive updates and new interactions.
- Reproducible experiments against strong baselines, including classical, sequential, graph, time-aware, and LLM4Rec baselines.
- Paper-ready tables, ablations, error analysis, and scripts.

## Repository map

```text
AGENTS.md                    durable Codex instructions
PROJECT_CHARTER.md           research thesis, novelty, risk plan
ROADMAP.md                   module-by-module project plan
TASKS.md                     actionable task queue for Codex runs
EXPERIMENTS.md               diagnostic and final experimental protocol
BASELINES.md                 baseline policy and reproduction requirements
CODEX_PROMPTS.md             prompts for worker/reviewer/lit-scout/experiment agents
PAPER_OUTLINE.md             living paper skeleton
configs/                     dataset, baseline, and experiment matrices
.codex/                      optional Codex project config and subagent roles
scripts/                     environment/data/experiment helper scripts
src/tglrec/                  package code: data, eval, graph, models, text, utils
tests/                       CLI, config, metrics, and preprocessing tests
```

## First non-toy milestone

The first milestone is not only an MVP. It must produce a falsifiable result:

1. preprocess MovieLens-1M and one Amazon sequential dataset with strict temporal splits;
2. reproduce at least SASRec, BERT4Rec, TiSASRec, and a popularity/item-kNN baseline;
3. run the three diagnostics: history shuffle, timestamp perturbation, and similarity-vs-transition candidate stress test;
4. generate a table showing whether the observed gain/loss is statistically meaningful;
5. log all configs and seeds.

Only after this diagnostic stage should the main model be optimized.

## Minimal local setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

GPU-dependent packages such as PyTorch, RecBole, or CUDA-specific wheels should be installed by following the official package instructions for the target machine. Do not pin a CUDA build in this repo until the target server is known.

## Runnable skeleton

Check that configs parse:

```bash
tglrec check-config configs/datasets.yaml
```

Run tests:

```bash
python -m pytest
```

On this Windows workspace the verified interpreter was Python 3.12:

```powershell
py -3.12 -m pytest -q
```

The current implemented package surface is CPU-only and covers deterministic config loading,
seed utilities, artifact manifests, ranking metrics, and MovieLens-1M preprocessing.

## MovieLens-1M preprocessing

Automatic download uses the official GroupLens archive checked on 2026-04-29:
https://files.grouplens.org/datasets/movielens/ml-1m.zip

```bash
tglrec preprocess movielens-1m --download --output-dir artifacts/datasets/movielens_1m
```

If network access to GroupLens is blocked, follow [DATA_MANUAL_STEPS.md](DATA_MANUAL_STEPS.md)
and run with a local archive or extracted directory:

```bash
tglrec preprocess movielens-1m --zip-path data/raw/movielens_1m/ml-1m.zip --output-dir artifacts/datasets/movielens_1m
tglrec preprocess movielens-1m --raw-dir data/raw/movielens_1m/ml-1m --output-dir artifacts/datasets/movielens_1m
```

The command writes:

```text
artifacts/datasets/movielens_1m/
  config.yaml
  metadata.json
  command.txt
  git_commit.txt
  checksums.json
  interactions.csv
  users.csv
  items.csv
  temporal_leave_one_out/{train,val,test}.csv
  global_time/{train,val,test}.csv
```

`temporal_leave_one_out` uses each user's last interaction as test and second-last interaction as
validation. `global_time` uses global timestamp cutoffs and validates that train timestamps precede
validation timestamps, which precede test timestamps.

`metadata.json` records same-user identical-timestamp tie statistics. MovieLens keeps those events
and breaks ties deterministically by stable item/raw ids. `metadata.json` also records SHA256
checksums for processed CSV files, and `checksums.json` fingerprints the full preprocessing
artifact manifest.

For time-sensitive evidence construction under leave-one-out, use the `training_events_as_of`
helper so another user's later training event is not treated as available before a target
prediction timestamp.

## Amazon Reviews 2023 local-file preprocessing

This repository does not download Amazon Reviews 2023 automatically. Follow
[DATA_MANUAL_STEPS.md](DATA_MANUAL_STEPS.md), keep the files outside git, and pass local category
files to the CLI:

```bash
tglrec preprocess amazon-reviews-2023 \
  --reviews-path data/raw/amazon_reviews_2023/All_Beauty.jsonl.gz \
  --metadata-path data/raw/amazon_reviews_2023/meta_All_Beauty.jsonl.gz \
  --category all_beauty \
  --output-dir artifacts/datasets/amazon_reviews_2023_all_beauty \
  --min-user-interactions 5 \
  --min-item-interactions 5 \
  --global-train-ratio 0.8 \
  --global-val-ratio 0.1 \
  --seed 2026
```

The loader accepts `.jsonl`, `.jsonl.gz`, `.json.gz`, `.ndjson`, and `.csv` files. By default it
uses `user_id`, `parent_asin` with `asin` fallback, `rating`, and auto-detects `timestamp` or
`unixReviewTime`. Timestamps are preserved as source integer values, and reviews are treated as
implicit positive interactions unless `--min-rating` is set. By default, repeated user-item events
after `parent_asin` normalization are collapsed to the first observed event so held-out targets are
not already in the user's history. Same-user identical timestamps are rejected by default because
their temporal order is ambiguous.

Full-horizon aggregate metadata fields such as `average_rating` and `rating_number` are excluded
from `items.csv` to avoid leaking future popularity/rating outcomes into item representations.
Outputs mirror MovieLens: `interactions.csv`, `users.csv`, `items.csv`,
`temporal_leave_one_out/`, `global_time/`, and manifest files with config, seed, raw file SHA256,
processed file SHA256, same-user timestamp tie statistics, split summaries, and provenance
metadata. The current `global_time` output is explicitly
full-horizon k-core/transductive; do not use it for inductive claims until train-period-only
filtering and ID mapping are implemented.

## CPU sanity baselines

Popularity and item-kNN/co-occurrence baselines can be evaluated without GPU dependencies:

```bash
tglrec evaluate sanity-baselines --dataset-dir artifacts/datasets/movielens_1m --output-dir runs/ml1m-sanity-baselines
```

The evaluator uses full-ranking over the processed item universe and filters previously seen
training items by default. Popularity and co-occurrence statistics are updated only from
`split=train` events with timestamps strictly before each held-out prediction event. For
`eval-split=test`, the user's validation event is added as prior user history by default but is not
added to global popularity/co-occurrence statistics; pass `--no-validation-history` to force a
train-only-history protocol. The CPU item-kNN path uses recent unique history windows by default;
set `--item-knn-max-history-items 0` and `--cooccurrence-history-window 0` to use all available
train-history items.

Each run writes:

```text
runs/<name>/
  config.yaml
  metrics.json
  metrics_by_segment.csv
  command.txt
  git_commit.txt
  stdout.log
  stderr.log
  environment.json
```

## CPU history perturbation diagnostics

The first diagnostic CLI slice evaluates `original`, `history_shuffle`, `order_reversal`,
`timestamp_removal`, `timestamp_randomization`, and `window_swap` for the CPU sanity baselines:

```bash
tglrec evaluate history-perturbations --dataset-dir artifacts/datasets/movielens_1m --output-dir runs/ml1m-history-perturbations
```

The perturbation protocol fixes the scorer's input event window first. Order perturbations shuffle
or reverse only that fixed window. Timestamp removal nulls history timestamps while preserving item
order, timestamp randomization permutes only observed pre-target history timestamps by seed/case,
and window swap exchanges within-week and long-gap observed history timestamps when both windows
exist. Popularity and the current item-kNN implementation are negative controls: they should be
invariant because they do not use sequence direction or timestamps once the history item multiset is
fixed. The command writes aggregate metrics, perturbation deltas, per-case paired target ranks/hit
deltas in `metrics_by_case.csv`, and segment aggregates. It does not yet implement
semantic-vs-transition stress labels.

## TDIG direct-transition graph

The first TDIG slice builds a deterministic directed item-transition graph from consecutive
per-user training events:

```bash
tglrec graph build-tdig --dataset-dir artifacts/datasets/movielens_1m --output-dir artifacts/graphs/ml1m-tdig
```

By default the builder uses only `split=train` rows from `interactions.csv` under
`temporal_leave_one_out`; pass `--split-name global_time` to build from the global-time training
split. Edges are directed `(source_item_id, target_item_id)` transitions and include support,
source-normalized transition probability, lift, PMI, first/last transition timestamp, mean gap,
time-gap bucket counts (`same_session`, `within_1d`, `within_1w`, `within_1m`, `long_gap`), and
direction asymmetry against the reverse edge.

For per-case leakage-safe candidate generation under leave-one-out, pass
`--strict-before-timestamp <prediction_timestamp>` or call
`build_tdig_from_events(..., strict_before_timestamp=...)` so train events from other users that
occur at or after the prediction time are excluded. A static train-only TDIG artifact is useful for
inspection and retrieval development, but it should not be treated as as-of-safe for paper-grade
per-target evaluation unless the evaluation protocol enforces the timestamp cutoff.

Directed transitions between same-user events with identical timestamps are skipped by default and
counted in `metadata.json` as `skipped_same_timestamp_transitions`. Use
`--include-same-timestamp-transitions` only for an explicit same-session ablation, because tied
timestamps do not establish a reliable temporal order.

The command writes:

```text
artifacts/graphs/<name>/
  edges.csv
  metadata.json
  config.yaml
  command.txt
  git_commit.txt
  created_at_utc.txt
  stdout.log
  stderr.log
  environment.json
  checksums.json
```

`metadata.json` records the SHA256 fingerprint of the source `interactions.csv` and, when present,
the processed dataset `checksums.json`, `config.yaml`, and `metadata.json` fingerprints. If the
processed dataset predates checksum manifests, the graph metadata records a provenance warning.

Python callers can use `tglrec.graph.build_tdig_from_events` for in-memory rows or
`tglrec.graph.build_tdig_from_processed_split` for processed dataset artifacts. Direct retrieval is
available through `TemporalDirectedItemGraph.retrieve_direct(source_item_id, top_k=...)` and returns
auditable edge statistics with deterministic tie-breaking. In-memory event rows without `event_id`
must not contain same-user same-timestamp ties; provide deterministic `event_id` values or resolve
the tie policy before graph construction.

## Compute policy

The project should start on CPU/small GPU with graph retrieval and lightweight reranking. Final SOTA-level experiments likely need at least one GPU for SASRec/BERT4Rec/TiSASRec/LLM-SRec-style baselines and dynamic GNN variants. API-based LLM calls are optional and should be restricted to small reranking/explanation studies unless the user explicitly provides budget and keys.
