# Phase 10 Master Plan: Observation To Full Recommender System

## Purpose

Phase 10 turns the current Phase 9E LoRA/control experiments into a complete
research system:

1. verify the observation that LLM-based recommenders may underuse temporal
   order and time gaps;
2. build our own time-aware graph-evidence framework;
3. compare against faithful official baselines under one protocol;
4. scale from current debugging domains to the four large same-candidate domains:
   `beauty`, `books`, `electronics`, and `movies`;
5. pass a top-conference reviewer gate before writing claims.

No result is paper evidence until it is produced from frozen artifacts,
prediction JSONL, saved metrics, diagnostics, and committed configs.

## Milestones

### M0: Observation Reproduction

Goal: make the motivating observation measurable before optimizing our method.

Required runs:

- fixed-label-mask `history_only_sft`;
- fixed-label-mask `temporal_evidence_sft`;
- sequence perturbation diagnostics: original, reversed, shuffled, recent-k;
- time-tag ablations: no time, absolute time, relative gap, bucketed gap;
- similarity-vs-transition candidate stress tests.

Exit gate:

- parse success and candidate adherence are sane;
- prompt-continuation diagnostics are logged;
- `limit=20` passes before any larger diagnostic;
- `limit=200` remains diagnostic only on `protocol_v1`;
- no conclusion is written from the current toy-ish protocol.

### M1: Our Framework

Goal: implement the original time-aware graph-evidence framework as reusable
modules, not as one-off scripts.

Framework components:

- temporal directed item graph construction from train-only interactions;
- time-window and time-decayed transition statistics;
- semantic-similarity and transition-need evidence separation;
- graph-to-language evidence translator;
- candidate-grounded LLM or local small-model reranker;
- optional temporal/dynamic graph encoder score channel;
- ablation switches for every evidence source.

Exit gate:

- every evidence artifact records train-only provenance;
- every prediction row preserves `event_id`, `source_event_id`, candidates,
  method, raw output, and metadata;
- ablations can be run by config rather than source edits;
- the framework beats smoke tests before server-scale runs.

### M2: Complete Recommendation System

Goal: compare our method against a full recommender stack, not only LLM prompts.

Required baseline families:

- non-personalized: random, popularity;
- traditional CF: item-kNN/BM25-like text ranking, BPR-MF or matrix
  factorization, LightGCN where feasible;
- sequential: SASRec plus at least one stronger sequential/time-aware baseline
  when feasible;
- text retrieval: BM25 and dense retrieval interface;
- LLM baselines: zero-shot/few-shot/rerank/constrained candidate rerank;
- official reference baselines: SLMRec, LLM-ESR, CLLM4Rec, RLMRec, and
  review-driven preference reasoning when their official code adaptation is
  faithful.

Fairness contract:

- shared split, candidates, event IDs, metric implementation, and prediction
  schema;
- shared Qwen3-8B backbone for LLM-based main-table baselines;
- shared project LoRA/QLoRA regime for that main LLM table;
- official/default or paper-recommended hyperparameters for baselines;
- validation-tuned TGL-Rec hyperparameters with logged search ranges and
  selected settings;
- each official baseline keeps its own algorithmic signal, adapted into the
  shared backbone/regime when possible;
- scaffold or non-official baselines stay out of main tables.

Exit gate:

- every reportable baseline has an implementation card, provenance, command,
  metrics, diagnostics, and reportability status;
- no table mixes protocols or candidate sets;
- paired comparison is possible from saved event IDs.

### M3: Four Large Domains

Goal: move paper claims from debugging data to the large same-candidate protocol.

Expected external task root:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Expected domains:

- `beauty`
- `books`
- `electronics`
- `movies`

Expected task pattern:

```text
{domain}_large10000_100neg_{valid,test}_same_candidate/
```

Protocol rules:

- do not resample users;
- do not resample negatives;
- do not regenerate candidates;
- preserve `event_id/source_event_id`, `user_id`, `item_id`, `split`, and exact
  candidate order;
- import under `protocol_week8_large10000_same_candidate` or a later explicit
  protocol version;
- keep `protocol_v1` as debugging history.

Exit gate:

- valid and test imported for every available domain;
- import manifests include external file checksums;
- readiness checks pass for all imported domains;
- every method runs on the same event/candidate rows.

### M4: Top-Conference Reviewer Gate

Goal: find and fix rejection reasons before paper writing.

Reviewer checks:

- novelty is not "G-Refer plus timestamps";
- observation is supported by perturbation experiments;
- our framework has a distinct mechanism, not prompt wording only;
- baselines are faithful and strong;
- official baselines are not weakened by generic local rewrites;
- no leakage from valid/test into graph evidence, SFT data, retrieval, or prompts;
- all results have seeds, configs, environment, git commit, and metrics files;
- statistical tests and paired comparisons are available;
- failure cases and limitations are documented.

Exit gate:

- a reviewer memo has no blocking P0/P1 issues;
- paper tables are generated from artifacts, not typed by hand;
- limitations are grounded in actual diagnostics.

## Server-First Workflow

Generate a server run plan:

```bash
python scripts/plan_four_domain_runs.py \
  --external-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output outputs/plans/four_domain_server_plan.json \
  --shell-output outputs/plans/four_domain_server_runbook.sh
```

Inspect the JSON and shell file before execution. The generated runbook does not
replace human judgment; it prevents forgetting import/check/eval order.

## Current Immediate Server Step

The two fixed-label-mask control adapters have reportedly trained successfully.
Before spending larger budget:

1. clear unrelated GPU processes;
2. run `scripts/run_lora_rerank_eval.py --limit 20 --top-m 50`;
3. run diagnostics only if `predictions.jsonl` exists;
4. proceed to larger diagnostic only if output behavior is sane.

## Files To Read First In A New Codex Thread

1. `docs/phase10_master_plan.md`
2. `docs/codex_handoff_phase9e.md`
3. `docs/week8_large_same_candidate_protocol.md`
4. `docs/reference_baseline_fidelity.md`
5. `docs/reference_method_adaptation_map.md`
6. `docs/method_card_time_graph_evidence.md`
