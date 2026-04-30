# Task queue

Use this as the living task board. Every completed task should leave a command, output path, and notes.

## Immediate tasks

### T0.1 Literature refresh

- Owner: lit_scout
- Output: `docs/literature_log.md`, updated `BASELINES.md`, updated `configs/baselines.yaml`
- Status: completed initial Module 0 refresh 2026-04-29
- Command/source process: lit_scout plus official web/source checks recorded in `docs/literature_log.md`
- Result: baseline sources and risks recorded for RecBole, MovieLens-1M, Amazon Reviews 2023, BPR, GRU4Rec, SASRec, BERT4Rec, TiSASRec, LightGCN, TGN, EvolveGCN, LLM-SRec, LLaRA, P5, ReLLa, G-Refer, G-CRS, FEARec, BSARec, and Meta generative recommenders.
- Notes: refresh again before final experiments; do not treat optional LLM/GNN entries as runnable until Module 2 validates licenses, dependencies, and split compatibility.

### T1.1 Package skeleton

- Owner: research_worker
- Output: source package, CLI stubs, tests
- Status: completed 2026-04-29
- Output: `src/tglrec/{data,eval,graph,models,text,utils}/`, `tglrec check-config`, metric/config/CLI tests
- Command: `py -3.12 -m pytest -q`
- Result: 12 passed
- Notes: CPU-only skeleton; no model framework dependencies installed.

### T1.2 MovieLens-1M preprocessing

- Owner: research_worker
- Output: processed interactions and metadata
- Status: completed for MovieLens-1M local/download ingestion 2026-04-29
- Output: `artifacts/datasets/movielens_1m/`
- Command: `py -3.12 -m tglrec.cli preprocess movielens-1m --download --download-dir data/raw/movielens_1m --output-dir artifacts/datasets/movielens_1m --min-user-interactions 5 --min-item-interactions 5 --seed 2026`
- Alternate command: `tglrec preprocess movielens-1m --zip-path data/raw/movielens_1m/ml-1m.zip --output-dir artifacts/datasets/movielens_1m`
- Result: real GroupLens download and preprocessing succeeded with 999611 interactions, 6040 users, and 3416 items after min-count filtering.
- Notes: writes `config.yaml`, `metadata.json`, `command.txt`, `git_commit.txt`, normalized CSVs, temporal leave-one-out splits, and global-time splits. Manual data fallback is in `DATA_MANUAL_STEPS.md`.

### T1.4 Leakage and metric guardrail follow-up

- Owner: reviewer + research_worker
- Status: partially completed 2026-04-29; strict as-of train statistics and preprocessing tie/count checksum guardrails are in place.
- Output: `training_events_as_of` helper, deterministic numeric metric tie-break helper, and explicit sanity-baseline `history_splits`.
- Done when: Module 4 graph builders and future candidate/negative samplers use strict as-of evidence, and same-timestamp policy is reflected in all downstream diagnostics.
- Notes: reviewer/repro audit found MovieLens-1M has many same-user equal-timestamp train/val/test ties. Current split is deterministic but not yet session-collapsed; fresh MovieLens preprocessing now records same-user same-timestamp tie statistics in `metadata.json` instead of rejecting the data. Amazon preprocessing keeps its default rejection policy and records tie statistics on successful outputs.

### T1.5 Processed dataset reproducibility manifests

- Owner: research_worker
- Status: completed 2026-04-30
- Output: `src/tglrec/data/artifacts.py`, preprocessing metadata updates, `checksums.json`
- Command: `py -3.12 -m pytest tests\test_movielens_preprocessing.py tests\test_amazon_preprocessing.py -q --basetemp .pytest_tmp\preprocess-guardrails`
- Result: 9 passed
- Notes: MovieLens and Amazon preprocessing now write processed CSV SHA256 entries into `metadata.json` and a full artifact checksum manifest in `checksums.json`. The manifest covers processed CSVs plus `config.yaml`, `metadata.json`, `command.txt`, `git_commit.txt`, and `created_at_utc.txt`. No network downloads were performed.

### T1.3 Amazon/Steam/Yelp preprocessing

- Owner: research_worker
- Output: at least one need-transition-heavy public dataset pipeline
- Status: partially completed for Amazon Reviews 2023 local-file ingestion 2026-04-29;
  Steam/Yelp remain pending.
- Output: `src/tglrec/data/amazon.py`, `tglrec preprocess amazon-reviews-2023`,
  `artifacts/datasets/amazon_reviews_2023_all_beauty/` when run with local raw files.
- Command: `python -m tglrec.cli preprocess amazon-reviews-2023 --reviews-path data/raw/amazon_reviews_2023/All_Beauty.jsonl.gz --metadata-path data/raw/amazon_reviews_2023/meta_All_Beauty.jsonl.gz --category all_beauty --output-dir artifacts/datasets/amazon_reviews_2023_all_beauty`
- Tests: `py -3.12 -m pytest tests\test_amazon_preprocessing.py -q --basetemp .pytest_tmp\amazon-tests` passed 3 tests.
- Done when: deterministic timestamped split tests mirror the MovieLens checks for Amazon plus
  at least one Steam or Yelp path if Module 1 still requires another domain.
- Notes: no large download or network path was added. Amazon loader accepts local JSONL/GZ/CSV,
  preserves source integer timestamps, uses `parent_asin` with `asin` fallback, writes metadata
  manifests/checksums/tie statistics, and validates temporal leave-one-out and global-time leakage constraints.

### T2.1 Sanity baselines

- Owner: research_worker
- Output: popularity and item-kNN metrics
- Status: completed CPU sanity run 2026-04-29
- Output: `src/tglrec/models/sanity_baselines.py`, `tglrec evaluate sanity-baselines`, `runs/20260429-ml1m-sanity-baselines-v2/`
- Command: `py -3.12 -m tglrec.cli evaluate sanity-baselines --dataset-dir artifacts/datasets/movielens_1m --output-dir runs/20260429-ml1m-sanity-baselines-v2 --ks 5 10 20 --item-knn-max-history-items 20 --cooccurrence-history-window 20`
- Result: full-ranking MovieLens-1M temporal LOO test over 6040 cases and 3416 items. Popularity HR@10=0.039238, NDCG@10=0.019279, MRR@10=0.013352. item-kNN HR@10=0.085596, NDCG@10=0.043738, MRR@10=0.031169.
- Tests: `py -3.12 -m pytest tests\test_metrics.py tests\test_sanity_baselines.py -q --basetemp .pytest_tmp\baseline-tests-2` passed 9 tests; `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-tests-2` passed 17 tests.
- Notes: global popularity/co-occurrence statistics are train-only and timestamp-strict. For `eval_split=test`, validation events are included only as the same user's prior history by default and recorded as `history_splits: [train, val_user_history_only]`. The v1 run under `runs/20260429-ml1m-sanity-baselines/` used the earlier train-only-history protocol and should be treated only as an obsolete preliminary artifact.
- Known blockers: this workspace is not a Git checkout, so run `git_commit.txt` contains `UNAVAILABLE: not a git repository`; do not cite these metrics as paper-grade claims until rerun from a real Git checkout. Existing dataset artifacts created before T1.5 should be regenerated to receive SHA256 manifests before final tables.
- User intervention required: none for CPU sanity work; Git checkout provenance is required before paper-grade experiment claims.
- Next recommended task: implement T1.3 Amazon Reviews 2023 local-file/download preprocessing, or add dataset checksum/tie-count metadata if prioritizing reproducibility hardening before the e-commerce dataset.

### T3.1 Diagnostic perturbations

- Owner: research_worker + reviewer
- Output: diagnostic CLI, tests, sample report
- Status: partially completed 2026-04-30 as a CPU sanity-baseline diagnostic suite with
  history-order and timestamp perturbation support. Per-case paired rank/delta logging was added
  2026-04-30. Semantic-vs-transition labels and paper-grade reruns remain pending.
- Output: `src/tglrec/eval/history_perturbations.py`, `tglrec evaluate history-perturbations`,
  `metrics_by_perturbation.csv`, `metrics_delta.csv`, `metrics_by_case.csv`, and
  `metrics_by_segment.csv` when run.
- Command: `py -3.12 -m tglrec.cli evaluate history-perturbations --dataset-dir artifacts/datasets/movielens_1m --output-dir runs/20260430-ml1m-history-perturbations-v2 --ks 5 10 20 --item-knn-max-history-items 20 --cooccurrence-history-window 20 --seed 2026`
- Result: MovieLens-1M full-ranking negative-control run over 6040 test cases. Popularity and
  item-kNN are invariant to `history_shuffle` and `order_reversal` after fixing the scoring
  history window before perturbation, as expected for order-invariant baselines. Treat
  `runs/20260430-ml1m-history-perturbations/` as obsolete because it was produced before this
  protocol correction and shows artificial item-kNN drops.
- Tests: `py -3.12 -m pytest tests\test_history_perturbations.py tests\test_sanity_baselines.py -q --basetemp .pytest_tmp\history-case-ranks-with-sanity` passed 7 tests.
  `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-history-perturbations-fixed` passed 26 tests.
- Latest slice 2026-04-30: added `timestamp_removal`, `timestamp_randomization`, and
  `window_swap` support to the fixed per-case scoring-history-event window. Timestamp removal
  nulls timestamps while preserving item order; timestamp randomization deterministically permutes
  only observed pre-target history timestamps by seed/case; window swap exchanges within-week and
  long-gap observed history timestamps while preserving item order and leaving cases without both
  windows unchanged. Output schemas remain stable; `config.yaml` now records the timestamp
  perturbation semantics.
- Tests for latest slice: `py -3.12 -m pytest tests\test_history_perturbations.py -q --basetemp .pytest_tmp\history-timestamp-perturbations`
  passed 5 tests; `py -3.12 -m pytest tests\test_history_perturbations.py tests\test_sanity_baselines.py -q --basetemp .pytest_tmp\history-timestamp-with-sanity-2`
  passed 10 tests.
- Final verification 2026-04-30: after aligning `window_swap` with within-week vs long-gap
  timestamp-window semantics, `py -3.12 -m pytest tests\test_history_perturbations.py -q --basetemp .pytest_tmp\history-timestamp-only-final`
  passed 6 tests; `py -3.12 -m pytest tests\test_history_perturbations.py tests\test_sanity_baselines.py -q --basetemp .pytest_tmp\history-timestamp-final-2`
  passed 11 tests; `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-history-timestamp-final`
  passed 30 tests.
- Notes: the default diagnostic now evaluates `original`, `history_shuffle`, `order_reversal`,
  `timestamp_removal`, `timestamp_randomization`, and `window_swap`. The model-input event window
  is fixed before perturbation. Train-only popularity and co-occurrence statistics remain
  timestamp-strict. Sensitivity index is reported as `(original - perturbed) / original` for
  nonzero original metrics. `metrics_by_case.csv` rows are deterministic and include `case_id`,
  `user_id`, `target_item_id`, `target_timestamp`, `model`, non-original perturbation,
  original/perturbed target rank, rank delta, and hit deltas for each configured K.
- Known blockers: semantic-vs-transition case labels remain `not_computed`; current MovieLens
  artifact lacks post-T1.5 `checksums.json`; this workspace is not a Git checkout, so
  `git_commit.txt` is `UNAVAILABLE`.
- User intervention required: none for CPU implementation. A real Git checkout plus regenerated
  dataset artifacts are required before paper-grade claims.
- Files changed in latest slice: `src/tglrec/eval/history_perturbations.py`,
  `src/tglrec/cli.py`, `tests/test_history_perturbations.py`, `README.md`, `TASKS.md`.
- Tests run in previous slice:
  `py -3.12 -m pytest tests\test_history_perturbations.py -q --basetemp .pytest_tmp\history-case-ranks`
  passed 2 tests;
  `py -3.12 -m pytest tests\test_history_perturbations.py tests\test_sanity_baselines.py -q --basetemp .pytest_tmp\history-case-ranks-with-sanity`
  passed 7 tests.
- Next recommended task: regenerate MovieLens artifacts with checksums before running another real
  diagnostic artifact, then add semantic-vs-transition case labels.
- Done when: shuffle/reversal/time perturbation metrics are available on one real dataset with
  versioned dataset artifacts and Git provenance.

### T4.1 TDIG graph builder

- Owner: research_worker
- Output: graph files and retrieval API
- Status: partially completed 2026-04-30 for deterministic direct-transition TDIG construction
  and retrieval API. Candidate recall measurement remains pending.
- Output: `src/tglrec/graph/tdig.py`, `tglrec graph build-tdig`,
  train-only `edges.csv` graph artifacts under `artifacts/graphs/<name>/` when run.
- Command: `py -3.12 -m tglrec.cli graph build-tdig --dataset-dir artifacts/datasets/movielens_1m --output-dir artifacts/graphs/ml1m-tdig`
- Tests: `py -3.12 -m pytest tests\test_tdig.py -q --basetemp .pytest_tmp\tdig`
  passed 5 tests; `py -3.12 -m pytest tests\test_tdig.py tests\test_cli.py tests\test_sanity_baselines.py -q --basetemp .pytest_tmp\tdig-with-cli`
  passed 13 tests; `py -3.12 -m pytest -q --basetemp .pytest_tmp\tdig-all`
  passed 36 tests. Final as-of guardrail verification:
  `py -3.12 -m pytest tests\test_tdig.py -q --basetemp .pytest_tmp\tdig-asof`
  passed 6 tests; `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-tdig-asof`
  passed 37 tests. Reviewer follow-up verification:
  `py -3.12 -m pytest tests\test_tdig.py -q --basetemp .pytest_tmp\tdig-review-fixes`
  passed 8 tests; `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-tdig-review-fixes`
  passed 39 tests. Repro-audit follow-up verification:
  `py -3.12 -m pytest tests\test_tdig.py -q --basetemp .pytest_tmp\tdig-repro-fixes`
  passed 9 tests; `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-tdig-repro-fixes`
  passed 40 tests; final verification `py -3.12 -m pytest -q --basetemp .pytest_tmp\all-final`
  passed 40 tests.
- Latest slice notes: direct edges are built from consecutive per-user training events only,
  sorted by user, timestamp, event id, and item id. Edge stats include support, source-normalized
  transition probability, lift, PMI, last/first/mean transition timestamp, mean gap,
  `same_session`/`within_1d`/`within_1w`/`within_1m`/`long_gap` histograms, and reverse-edge
  direction asymmetry. Retrieval tie-breaking is deterministic. The processed-split builder and CLI
  now expose `strict_before_timestamp` / `--strict-before-timestamp` for per-case as-of graph
  construction; static train-only artifacts are for inspection or protocols whose split already
  enforces the required global time cutoff. Reviewer blockers addressed: missing-`event_id`
  same-user same-timestamp ties now raise an explicit error, and gap-bucket direct retrieval filters
  out zero-support edges for the requested bucket. Repro-auditor follow-ups addressed in code:
  TDIG metadata records input dataset fingerprints/provenance warnings, same-user identical
  timestamp directed transitions are skipped by default and counted, and `environment.json`
  includes key package versions.
- Known blockers: GitHub publication is blocked in this workspace because `gh` is not installed,
  `git ls-remote https://github.com/appleweiping/TGL-Rec.git` fails with Windows credential error
  `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`, and both normal `git init -b main`
  and fallback `git init --bare --initial-branch=main git-upload.git` fail while writing Git
  `config.lock`/`config` with `Permission denied`. Rerun/push from an authenticated GitHub
  environment with writable Git metadata. Existing graph artifacts created before the repro fixes
  should be regenerated before use.
- Next recommended task: add TDIG candidate recall evaluation under a global-time split or an
  explicit per-case `strict_before_timestamp` cache, then connect similarity-vs-transition labels.
- Real artifact smoke: `py -3.12 -m tglrec.cli graph build-tdig --dataset-dir artifacts/datasets/movielens_1m --output-dir artifacts/graphs/ml1m-tdig`
  wrote `edges.csv` with 500422 edges and 981491 transitions plus metadata and checksums. The
  command printed successful output but the shell wrapper timed out after 134 seconds, so rerun with
  a longer timeout before treating the runtime log as paper-grade.
- Files changed: `src/tglrec/graph/tdig.py`, `src/tglrec/graph/__init__.py`, `src/tglrec/cli.py`,
  `tests/test_tdig.py`, `README.md`, `TASKS.md`.
- Done when: direct transition retrieval is tested and candidate recall is reported.

## Mid-stage tasks

### T2.2 Strong sequential baselines

- Implement/integrate SASRec, BERT4Rec, TiSASRec.
- Make sure splits and negatives are identical.
- Tune with comparable budget.

### T3.2 Similarity-vs-transition stress test

- Build semantic item embeddings from item text.
- Generate hard candidates.
- Report Semantic Trap Rate and Transition Win Rate.

### T5.1 Graph-to-language evidence

- Implement deterministic templates.
- Export structured evidence JSON and text.
- Add tests that all evidence maps to graph stats.

### T6.1 Need-aware gate

- Build feature extraction and reranker.
- Run ablations on MovieLens-1M and one Amazon category.

## Final-stage tasks

### T7.1 Dynamic GNN channel

- Implement or integrate TGN-style memory.
- Evaluate global-time split and cold/new item segments.

### T8.1 Main tables

- Run all final baselines and model variants.
- Generate LaTeX tables and plots.

### T8.2 Reviewer attack pass

- Spawn reviewer/repro_auditor/lit_scout.
- Write `docs/reviewer_attack.md` with likely reviewer objections and responses.

### T8.3 Paper draft

- Fill `PAPER_OUTLINE.md` with actual results.
- Do not invent missing claims.
