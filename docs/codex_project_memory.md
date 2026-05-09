# Codex Project Memory

This is the first document every future Codex thread should read before doing
nontrivial work in this repository. Keep it current. If this document becomes
stale, later agents will make stale plans.

## Read First

Before edits, read these files in order for any nontrivial task:

1. `docs/codex_project_memory.md`
2. `docs/phase10_master_plan.md`
3. `docs/server_runbook.md`
4. `docs/week8_large_same_candidate_protocol.md`
5. `docs/reference_baseline_fidelity.md`
6. `docs/reference_method_adaptation_map.md`
7. `docs/method_card_time_graph_evidence.md`

If the task touches results, claims, baselines, or method novelty, also read:

- `docs/reference_baseline_fidelity.md`
- `docs/reference_implementation_cards/`
- `docs/week8_large_same_candidate_protocol.md`
- `docs/reproducibility.md`
- `docs/paper_table_plan.md`
- `docs/literature_log.md`

`docs/codex_handoff_phase9e.md` is historical Phase 9E context. Use it for
provenance, not as the active plan when it conflicts with this memory or the
Phase 10 master plan.

## Current Direction

The project is Phase 10: observation to original framework to full recommender
system.

The research path is:

1. Measure the observation that LLM recommenders may rely on semantic similarity
   and popularity more than temporal need transitions.
2. Build our original TGL-Rec framework around temporal directed item graph
   evidence, graph-to-language translation, need-aware gating, and
   candidate-grounded local Qwen3-8B reranking.
3. Compare against strong, faithful baselines under one frozen protocol.
4. Scale from diagnostic `protocol_v1` data to four large same-candidate domains:
   `beauty`, `books`, `electronics`, and `movies`.
5. Run reviewer and reproducibility gates before paper writing.

Do not reduce this to a prompt-only tweak, generic LoRA SFT, or toy demo. The
method must have distinct mechanisms, ablations, provenance, and failure-case
analysis.

Two milestones are now the most important project checkpoints:

### Large-Scale Observation Milestone

Observation should run on the same scale as later training whenever feasible:
full `beauty` plus `books`, `electronics`, and `movies` with 10,000 users per
domain under the same-candidate protocol. The observation matrix should include:

- base Qwen3-8B reranking/inference with no adapter;
- the fixed-label-mask history-only and temporal-evidence controls;
- at least four senior-reference Qwen3-8B adapted baseline probes once faithful
  official-code probe wrappers exist;
- sequence perturbation, time-tag, similarity-vs-transition, parse/adherence,
  and candidate-grounding diagnostics.

This milestone is complete only when the pain point seen in base Qwen3-8B is
checked against faithful senior-reference probes, not only against our own
control prompts.

### Formal Baseline Milestone

After the observation identifies the correctable pain point, each baseline must
be rebuilt as a formal, fair baseline under the shared protocol. Formal means:

- official algorithm preserved;
- Qwen3-8B/project LoRA or QLoRA policy applied where faithful;
- same data, candidates, splits, metric code, prediction schema, and event IDs;
- baseline official/default hyperparameters recorded;
- our method validation tuning recorded separately;
- paired statistics, diagnostics, and exportable tables generated from
  artifacts.

No `reference_*_sft` scaffold may satisfy this milestone.

## Senior Baseline Advice Adopted

The main LLM baseline setting follows the senior-advised academic fairness
choice:

- get source code from the official implementation whenever possible;
- run all reportable LLM baselines on our frozen data and candidate protocol;
- use Qwen3-8B as the shared base model/backbone where faithful;
- use the project LoRA/QLoRA regime for the main LLM table;
- preserve each baseline's official algorithm, losses, heads, adapters,
  evidence construction, and scoring logic as much as possible;
- use official/default or paper-recommended hyperparameters for baselines;
- tune TGL-Rec on validation data, with search ranges and selected settings
  logged;
- evaluate all methods with the same split, candidates, metrics, prediction
  schema, and paired event IDs.

Do not claim equal-budget tuning unless it was actually run. Do not call a
generic local rewrite an official baseline. If official code is missing, mark the
method as blocked or non-official until the user explicitly approves a labeled
reproduction.

Current official-code baseline queue:

- `slmrec_distill_qwen_lora`
- `llm_esr_qwen_lora`
- `cllm4rec_qwen_lora`
- `rlmrec_qwen_lora`
- `review_pref_reasoning_qwen_lora`

Blocked from main tables unless official code is found or the user approves a
non-official reproduction:

- `controllable_rec_qwen_lora`
- `transrec_qwen_lora`

The `reference_*_sft` scaffold variants are not reportable senior baselines by
themselves. They are implementation scaffolds and controls only.

## Data And Protocol Memory

`protocol_v1` with MovieLens and the filtered Amazon debug set is useful for
debugging and diagnostics, but it is too toy-like for final paper claims.

The intended large protocol comes from the adjacent server project:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Expected directories:

```text
{domain}_large10000_100neg_{valid,test}_same_candidate/
```

Expected domains:

- `beauty`
- `books`
- `electronics`
- `movies`

There is also an older adjacent/local data source that future agents must
remember:

```text
D:/Research/Uncertainty-LLM4Rec/data/processed
~/projects/uncertainty-llm4rec/data/processed
```

It may contain messy domain directories such as `amazon_beauty`,
`amazon_books`, `amazon_electronics`, `amazon_movies`, and
`processed_4domains/amazon_*_small`. These are not automatically the final
Week8 1+100 same-candidate protocol. Treat them as
`protocol_uncertainty_legacy_local` diagnostic/local sources unless a manifest
proves they match the final protocol. Use
`scripts/normalize_adjacent_same_candidate.py` to stage ranking files from this
source before importing through the Week8-compatible importer.

Rules:

- do not resample users;
- do not resample negatives;
- do not alter candidate order;
- preserve `event_id`, `source_event_id`, `user_id`, `item_id`, and `split`;
- keep paired comparison possible for every prediction row;
- import under an explicit protocol version such as
  `protocol_week8_large10000_same_candidate`;
- keep `protocol_v1` as diagnostic history only.

## Our Framework Must Stay Original

The TGL-Rec contribution should be stronger than a stitched collection of
existing ideas. Preserve and deepen these mechanisms:

- temporal directed item graph built from train-only interactions;
- time-windowed and time-decayed transition statistics;
- semantic-similarity vs temporal-transition separation;
- need gate that decides when graph evidence should matter;
- temporal need-state channel for stable preference, transition pressure, user
  drift, semantic traps, and evidence confidence;
- graph-to-language evidence translator with auditable factor scores;
- candidate-grounded LoRA/local reranking path;
- optional temporal graph encoder score channel;
- ablations for evidence source, gate, temporal windows, semantic trap penalty,
  contrastive evidence, and reranker choice.

Every mechanism should be configurable, logged, and evaluated through the shared
prediction schema. Do not hard-code model paths, dataset paths, prompts, seeds,
or protocol details in source files.

When researching or adapting ideas, future agents may carefully read and
understand senior-recommended papers, official projects, and other top-tier
conference systems. They are only references for understanding method design.
Our actual contribution cannot be stitched, copied, or presented as a
recombination of those methods. If a design element is inspired by prior work,
document what is borrowed as context, what is different in TGL-Rec, and which
ablation proves the difference matters.

In short: TGL-Rec cannot be stitched, copied, or presented as a recombination of
senior-reference or other top-tier methods.

## Server Collaboration Protocol

Codex runs locally and cannot inspect the shared server directly. The user runs
server commands and pastes logs or errors back.

Therefore:

- give exact server commands when server work is needed;
- do not imply a server command succeeded unless the user pasted evidence;
- when logs show failure, diagnose from the pasted output and update the next
  command;
- before long GPU jobs, include checks such as `nvidia-smi`, output-directory
  preservation, and `test -f`/`test -d` guards;
- preserve old outputs by moving them to timestamped directories before reruns;
- never commit private server configs, model checkpoints, PDFs, output artifacts,
  or copied paper text.

When commands change, update `docs/server_runbook.md` so the next agent does not
repeat stale instructions.

## Multi-Agent Workflow

For every complex task, use multi-agent collaboration when tools are available.
Complex means implementation beyond a narrow one-file fix, experiment design,
baseline adaptation, literature update, method design, server run planning,
paper-claim review, or anything that changes reportability. A reasonable
default split is:

- implementation worker for bounded code changes;
- reviewer for fairness, leakage, and correctness;
- reproducibility auditor for configs, seeds, artifacts, and run commands;
- literature scout for official code, paper details, and baseline provenance.

If the agent tool is unavailable or blocked by thread limits/rate limits, state
that explicitly and continue with a self-review checklist. Do not use multi-agent
overhead for tiny one-command or one-line tasks.

For complex research tasks, at least one review pass must compare TGL-Rec's
rigor, novelty, technical depth, and component completeness against multiple
top-conference papers or official projects, not only the few current senior
references. Search or inspect current literature when this comparison depends on
up-to-date papers, code availability, or benchmark practice. The comparison must
not copy methods; it is a pressure test for whether our project is deep enough.
Use multiple top-conference papers or official projects as the comparison set,
not a single convenient baseline.

## Ending Criteria

The project and experiment stage can be called basically complete only when all
of these are true:

- large-scale observation has run on the frozen four-domain protocol or a
  documented final replacement protocol;
- at least four faithful official/senior baselines are implemented, or any
  missing ones are blocked with reviewer-acceptable reasons and replacements;
- TGL-Rec has reportable configs, ablations, diagnostics, paired statistics, and
  exported tables from saved prediction artifacts;
- leakage, reproducibility, candidate-alignment, and significance checks pass;
- a top-conference-style reviewer pass finds no blocking P0/P1 issues in
  novelty, fairness, rigor, or technical depth;
- failure cases and limitations are documented from real diagnostics.

Only then should Codex tell the user that the project/experiment phase is
basically complete and ready for paper writing. Until then, each final response
after a complex task must state the next concrete plan and the remaining gate.

## Completion Contract

After every meaningful stage, update the durable docs before finalizing:

- `docs/codex_project_memory.md` if direction, policy, baseline status, or next
  server workflow changed;
- `docs/phase10_master_plan.md` if milestones or gates changed;
- `docs/server_runbook.md` if runnable server commands changed;
- baseline implementation cards if a baseline status changed;
- audit/handoff docs if new real results, errors, or server outcomes were
  reported by the user.

Then run targeted tests, report what passed or failed, commit the changes, and
push the branch so the server can `git pull`. If push is impossible, say exactly
why and what remains local.

Never fabricate results, logs, metrics, tables, conclusions, or claims. Paper
writing starts only after real artifacts exist.

At the end of every complex task, provide a concise completion note with:

- what changed;
- what was tested;
- whether the task is complete or what remains blocked;
- the next step or plan;
- the current project/experiment gate toward ending the experimental phase.
