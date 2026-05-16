# Phase 10 Master Plan: Observation To Full Recommender System

## Purpose

Phase 10 turns the current Phase 9E LoRA/control experiments into a complete
research system:

1. verify the observation that LLM-based recommenders may underuse temporal
   order and time gaps;
2. build our own time-aware graph-evidence framework;
3. compare against the Pony/Uncertainty official same-candidate baseline suite
   under one protocol instead of rebuilding a separate baseline queue;
4. scale from current debugging domains to the four large same-candidate domains:
   `beauty`, `books`, `electronics`, and `movies`;
5. pass a top-conference reviewer gate before writing claims.

No result is paper evidence until it is produced from frozen artifacts,
prediction JSONL, saved metrics, diagnostics, and committed configs.

For durable project memory and future Codex behavior, read
`docs/codex_project_memory.md` first. It records the senior baseline advice,
server relay protocol, multi-agent workflow, and update/commit/push contract.

## Milestones

### M0: Observation Reproduction

Goal: make the motivating observation measurable before optimizing our method.

Required runs:

- large-scale base Qwen3-8B observation on full `beauty` plus 10,000-user
  `books`, `electronics`, and `movies` same-candidate tasks when available;
- fixed-label-mask `history_only_sft`;
- fixed-label-mask `temporal_evidence_sft`;
- Pony official baseline score/provenance reuse for cross-method diagnostics
  once exact same-candidate score gates pass;
- sequence perturbation diagnostics: original, reversed, shuffled, recent-k;
- time-tag ablations: no time, absolute time, relative gap, bucketed gap;
- similarity-vs-transition candidate stress tests.

Exit gate:

- parse success and candidate adherence are sane;
- prompt-continuation diagnostics are logged;
- `limit=20` passes before any larger diagnostic;
- `limit=200` remains diagnostic only on `protocol_v1`;
- four-domain observation uses the same candidate/event rows as later formal
  training and evaluation;
- Pony official baselines are reused through manifest/provenance/score-gate
  checks rather than approximated with `reference_*_sft` scaffolds;
- no conclusion is written from the current toy-ish protocol.

### M0.5: Large-Scale Observation Matrix

Goal: verify that the base Qwen3-8B pain point is not an artifact of one prompt
or one small protocol.

Required observation matrix:

- domains: full `beauty`, plus `books`, `electronics`, and `movies` with 10,000
  users per domain when available;
- candidates: one positive plus 100 negatives from the frozen same-candidate
  external tasks;
- methods: base Qwen3-8B, history-only control, temporal-evidence control, and
  Pony official baselines whose score/provenance artifacts pass exact gates;
- diagnostics: parse success, candidate adherence, hallucination, sequence
  perturbation, time-tag ablation, similarity-vs-transition stress cases,
  per-domain and aggregate summaries.

Exit gate:

- base Qwen3-8B observation runs from a server command with old outputs
  preserved;
- Pony official baselines are either reusable with passing gates or explicitly
  pending/blocked, such as `promax` pending and `setrec` replaced;
- every prediction row preserves `event_id/source_event_id`;
- no observation output is merged into final paper tables unless later promoted
  under the formal baseline gate.

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

### M2: Complete Recommendation System With Pony Official Baselines

Goal: compare our method against a full recommender stack, not only LLM prompts.

The active main baseline suite is reused from Pony/Uncertainty because it uses
the same data selection, same-candidate event rows, Qwen3-8B declared-adaptation
policy, and official-code/official-code-level provenance.

Active completed candidates:

- `llm2rec`
- `llmesr`
- `llmemb`
- `rlmrec`
- `irllrec`
- `elmrec`
- `proex`

Planned pending candidate:

- `promax`, the last 2026 official baseline, excluded from completed main
  tables until all declared domains pass exact-score gates.

Blocked/replaced:

- `setrec`, blocked by upstream large-domain failure and replaced by `elmrec`,
  `proex`, and `promax`.

Fairness contract:

- shared split, candidates, event IDs, metric implementation, and prediction
  schema;
- shared Qwen3-8B backbone for LLM-based main-table baselines;
- method-declared adapter, representation, graph, intent, or profile modules
  retained when they are part of the official algorithm;
- official/default or paper-recommended hyperparameters for baselines;
- validation-tuned TGL-Rec hyperparameters with logged search ranges and
  selected settings;
- each official baseline keeps its own algorithmic signal, adapted into the
  shared same-candidate protocol when possible;
- scaffold or non-official baselines stay out of main tables.

Exit gate:

- every reused Pony baseline has manifest status, official repo, pinned commit,
  evidence/archive path, score-gate status, metrics, diagnostics, and
  reportability status;
- no table mixes protocols or candidate sets;
- paired comparison is possible from saved event IDs.

### M2.5: Pony Baseline Reuse And Migration

Goal: make the Pony official baseline system a first-class TGL-Rec baseline
source without rerunning already completed baselines or copying large artifacts
into git.

Rules:

- `configs/baselines/pony_official_external.yaml` is the active manifest;
- Pony score files must use `source_event_id,user_id,item_id,score`;
- exact key match to frozen candidates, no missing/extra/duplicate keys, and
  finite scores are required before import;
- baseline default/recommended hyperparameters are logged from Pony provenance;
- TGL-Rec validation tuning is logged separately;
- leakage audits and paired statistical comparisons run before table export.

Exit gate:

- completed Pony baselines are imported or linked with provenance and exact
  same-candidate score audits;
- `promax` is either completed across all declared domains or clearly excluded
  from completed main tables as pending;
- no `reference_*_sft` scaffold is used as a main-table baseline;
- the second migration stage has a TGL-Rec-side runner/importer plan, but the
  first reset stage does not copy large evidence archives into git.

### M3: Four Large Domains

Goal: move paper claims from debugging data to the large same-candidate protocol.

Expected external task root:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Expected frozen external task families:

```text
beauty_supplementary_smallerN_100neg_{valid,test}_same_candidate/
books_large10000_100neg_{valid,test}_same_candidate/
electronics_large10000_100neg_{valid,test}_same_candidate/
movies_large10000_100neg_{valid,test}_same_candidate/
```

Protocol rules:

- do not resample users;
- do not resample negatives;
- do not regenerate candidates;
- do not edit `candidate_items.csv`, `ranking_valid.jsonl`, or
  `ranking_test.jsonl`;
- preserve `event_id/source_event_id`, `user_id`, `item_id`, `split`, and exact
  candidate order;
- every method must export scores as `source_event_id,user_id,item_id,score`;
- import evaluation scores through `main_import_same_candidate_baseline_scores.py`;
- do not use test split for hyperparameter selection;
- official LLM2Rec result reuse is limited to scores, provenance, and audit
  artifacts;
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
- the method is not stitched, copied, or presented as a recombination of
  senior-reference papers;
- rigor, novelty, technical depth, and component completeness have been compared
  against multiple top-conference papers/projects, not only the immediate senior
  references;
- observation is supported by perturbation experiments;
- our framework has a distinct mechanism, not prompt wording only;
- baselines are faithful and strong;
- Pony official baselines are not weakened or replaced by generic local rewrites;
- no leakage from valid/test into graph evidence, SFT data, retrieval, or prompts;
- all results have seeds, configs, environment, git commit, and metrics files;
- statistical tests and paired comparisons are available;
- failure cases and limitations are documented.

Exit gate:

- a reviewer memo has no blocking P0/P1 issues;
- paper tables are generated from artifacts, not typed by hand;
- limitations are grounded in actual diagnostics.

## Experiment Ending Gate

Do not let the project drift into endless "next steps." The experiment phase can
be considered basically complete, and paper writing can begin, only after:

- the large-scale observation matrix is complete on the frozen four-domain
  protocol or a documented final replacement;
- TGL-Rec's reportable framework and ablations are implemented and run;
- the Pony official baseline suite is reused/migrated with exact score gates,
  and pending `promax` status is explicit if not complete;
- paired statistics, leakage checks, reproducibility checks, and table exports
  are generated from saved artifacts;
- a top-conference-style reviewer pass finds no blocking P0/P1 issue in novelty,
  fairness, rigor, technical depth, or evidence.

Every complex-task final report should say which gate remains before the
experiment phase can end.

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

The four-domain generated plan now also includes:

- `observation_qwen3_base`: executable non-reportable base Qwen3-8B observation
  smoke run using `configs/experiments/week8_qwen3_8b_base_observation.yaml`;
- `pony_official_baseline_reuse`: planned reuse checks for Pony official
  same-candidate score/provenance artifacts;
- `pony_official_pending_baselines`: planned pending entries such as `promax`
  until all declared domains pass exact-score gates;
- `ours_framework_ablation_matrix`: planned ablations, blocked until reportable
  Phase 10 framework configs exist.

## Multi-Agent And Update Workflow

Use multi-agent collaboration for every complex implementation, baseline
adaptation, experiment design, literature checking, or reviewer-gate work when
tools are available. Pair implementation with review/reproducibility checks
instead of relying on one linear pass. If agent tools are unavailable, record the
blocker and perform an explicit self-review.

After each meaningful stage, update the durable memory, this plan, the server
runbook, and affected baseline cards if next steps, commands, baseline status,
or real server outcomes changed. Commit and push completed local work so the
server can continue with `git pull`. Final reports after complex tasks must
include a concrete next-step plan and the current gate toward ending the
experiment phase.

## Files To Read First In A New Codex Thread

1. `docs/codex_project_memory.md`
2. `docs/phase10_master_plan.md`
3. `docs/server_runbook.md`
4. `docs/week8_large_same_candidate_protocol.md`
5. `docs/reference_baseline_fidelity.md`
6. `docs/reference_method_adaptation_map.md`
7. `docs/method_card_time_graph_evidence.md`

`docs/codex_handoff_phase9e.md` is historical context. Do not let it override
the current Phase 10 plan.
