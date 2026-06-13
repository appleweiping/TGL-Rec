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
3. Compare against the Pony/Uncertainty official same-candidate baseline suite
   under the shared frozen protocol instead of rebuilding a separate baseline
   queue from scratch.
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
- Pony official baseline rows whose score/provenance artifacts pass exact
  same-candidate gates;
- sequence perturbation, time-tag, similarity-vs-transition, parse/adherence,
  and candidate-grounding diagnostics.

This milestone is complete only when the pain point seen in base Qwen3-8B is
checked against Pony official baseline behavior, not only against our own
control prompts.

### Pony Official Baseline Reuse/Migration Milestone

After the observation identifies the correctable pain point, TGL-Rec should
reuse and migrate the Pony/Uncertainty official baseline system because it was
run by the same owner on the same data selection and same-candidate design.
Formal means:

- official algorithm or official-code-level implementation preserved in Pony;
- Qwen3-8B declared-adaptation policy applied where the baseline consumes LLM or
  text representations;
- same data, candidates, splits, metric code, score schema, and event IDs;
- baseline official/default or recommended hyperparameters recorded;
- our method validation tuning recorded separately;
- paired statistics, diagnostics, and exportable tables generated from imported
  Pony score/provenance artifacts plus TGL-Rec method artifacts.

The active manifest is `configs/baselines/pony_official_external.yaml`. No
`reference_*_sft` scaffold may satisfy this milestone.

## Pony Official Baseline Policy

The main baseline setting follows the Pony official-code fairness policy:

- reuse Pony official-code or official-code-level baselines already run on the
  frozen same-candidate protocol;
- preserve the official implementation path or official-code-level provenance
  whenever a Pony baseline enters the main table;
- use Qwen3-8B as the shared LLM/text backbone where the baseline requires an
  LLM or text representation;
- preserve each baseline's official algorithm, losses, heads, adapters,
  representation modules, graph/intent/profile components, and scoring logic as
  much as possible;
- use official/default or recommended hyperparameters for baselines;
- tune TGL-Rec on validation data, with search ranges and selected settings
  logged;
- keep the project LoRA/QLoRA regime for TGL-Rec controls and any future
  method-declared adapter work;
- evaluate all methods with the same split, candidates, metrics,
  `source_event_id,user_id,item_id,score` score schema, and paired event IDs.

Do not claim equal-budget tuning unless it was actually run. Do not call a
generic local rewrite an official baseline. Do not rerun or replace the Pony
baseline suite with a new unrelated queue unless the user explicitly changes
the paper strategy.

Active Pony official baseline suite:

- completed main-table candidates (all 8, all 4 domains): `llm2rec`, `llmesr`,
  `llmemb`, `rlmrec`, `irllrec`, `elmrec`, `proex`, and `promax`;
- blocked/replaced: `setrec`, replaced by `elmrec`, `proex`, and `promax`.

The old `reference_*_sft` scaffold variants are historical/non-reportable
containers only. They are not the active main baseline plan.

## Data And Protocol Memory

`protocol_v1` with MovieLens and the filtered Amazon debug set is useful for
debugging and diagnostics, but it is too toy-like for final paper claims.

The intended large protocol comes from the adjacent server project:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Current frozen external task families:

```text
beauty_supplementary_smallerN_100neg_{valid,test}_same_candidate/
books_large10000_100neg_{valid,test}_same_candidate/
electronics_large10000_100neg_{valid,test}_same_candidate/
movies_large10000_100neg_{valid,test}_same_candidate/
```

Expected domains:

- `beauty`: supplementary smaller-N 100-negative package;
- `books`: 10,000-user 100-negative package;
- `electronics`: 10,000-user 100-negative package;
- `movies`: 10,000-user 100-negative package.

This package is a reusable evaluation protocol artifact. It is not model
weights and not a paper result by itself. It contains frozen same-candidate
evaluation inputs such as `ranking_valid.jsonl`, `ranking_test.jsonl`,
`candidate_items.csv`, and `train_interactions.csv`.

Rules:

- do not resample users;
- do not resample negatives;
- do not edit `candidate_items.csv`, `ranking_valid.jsonl`, or
  `ranking_test.jsonl`;
- do not alter candidate order or candidate membership;
- every method score file must use the shared schema
  `source_event_id,user_id,item_id,score`;
- import baseline/model scores for evaluation through
  `main_import_same_candidate_baseline_scores.py`;
- do not use the test split for hyperparameter selection;
- preserve `event_id`, `source_event_id`, `user_id`, `item_id`, and `split`;
- keep paired comparison possible for every prediction row;
- import under an explicit protocol version such as
  `protocol_week8_large10000_same_candidate`;
- keep `protocol_v1` as diagnostic history only.
- reuse Pony official baseline score/provenance artifacts only after exact
  same-candidate score-gate checks pass;
- do not copy large Pony `.tar.gz` evidence archives into git. Record paths,
  hashes or sizes, summaries, and import status instead.

If reusing official LLM2Rec results from the adjacent project, only reuse
scores, provenance, and audit artifacts. Do not make intermediate checkpoints
or embeddings a long-term required artifact for TGL-Rec.

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

**Local is primary. Server is experiment-only.**

- All code, docs, git commit/push happen locally (D:\research\TGL-Rec)
- Server only does: `git pull` → execute experiments → output results
- Never write code or commit from server
- GitHub updates always come from local push

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
- Pony official baseline reuse/migration is complete for the active suite, with
  any pending baseline such as `promax` clearly marked until all domains pass;
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

## 2026-06-12 — CC-PACE data seams wired; beauty zero-shot GO pair running

- Data seams DONE (branch `feat/cc-pace-data-seams`): `build_cc_pace_cf_artifacts.py`
  (frozen SASRec -> panel-z scores / neighbour titles / KMeans clusters / train pop /
  title-lexicon category), `build_cc_pace_profiles.py` + `methods/cc_pace/text_facets.py`,
  driver `--cf-artifacts/--profiles`, runner bootstraps the frozen task file from Pony.
  Coverage audit: positives 95.1% / negatives 98.5% in train vocab (no seen-ness leak).
- `hf_judge.py` rewritten for feasibility: shared prefill + bounded-batch (B=8) label
  scoring with sequential OOM fallback; 19.5s/user measured (was 71s sequential; the
  original per-label full-forward was months-infeasible). Equivalence tests (5) green —
  they caught transformers 5.x `batch_repeat_interleave` returning None (in-place) and a
  46.8GB cache-materialization OOM. Driver checkpoints per-user metrics (resume + paired
  bootstrap input).
- LoRA driver ready: `train_cc_pace_lora.py` (single-digit-token PL surrogate so all 101
  scores live in one graph; prefix-trained SASRec for train-panel CF — pseudo-positive
  excluded; fold A/B saved for split-conformal; torch dCor parity-tested). GO verdict
  script: `cc_pace_go_verdict.py`.
- RUNNING on server (queue `~/projects/gpu_queue_20260612.sh`): text_only 973 -> full 973
  (resume) -> then TRUCE-Rec Stage-B (separate project, same GPU, serial). Early running
  NDCG@10 at 100 users: full ~0.105-0.124 band (noisy; GO bar 0.13 needs full n).
- Server sync = git bundle over scp (server cannot reach GitHub; TLS reset). Server env:
  `tglrec-lora` (documented `tglrec` env does not exist). Pushes go to the feature branch
  (direct main pushes are policy-blocked locally); merge to main via PR/user.
- Next: when full+text_only finish -> `cc_pace_go_verdict.py`; GO -> LoRA train + adapter
  eval vs 0.1506 (paired bootstrap) + remaining 3 zero-shot ablation variants; NOT GO ->
  3-seat ARIS redesign per CLAUDE.md rule 9. Gate: performance table before any paper text.

## 2026-06-13 — Beauty full OOM diagnosed; HF judge recovery patch prepared

- Server truth: `text_only` completed 973 users with NDCG@10 = 0.1108. `full` resumed only
  65/973 users, then failed before writing `full.json`; `go_verdict.json` is missing, so the
  GO/KILL decision has NOT happened and LoRA must NOT start.
- Failure: Qwen3-8B label scoring OOMed while materializing expanded prompt KV cache in
  `hf_judge._expand_past`; the old OOM fallback immediately retried singleton scoring with
  shared `past` and `use_cache=True`, which could append to the shared prompt cache before crop.
- Local patch: `hf_judge` now uses a read-only expanded-cache wrapper after the first safe-resume
  attempt still peaked near 48GB. The wrapper avoids Transformers `DynamicCache.update` materializing
  and retaining prompt-sized cache copies across all decoder layers; the scorer still supports
  adaptive suffix-batch halving, clears CUDA cache fragments after each panel, and keeps the shared
  prefill cache immutable. Added CPU/server tests for simulated OOM fallback and a
  `cc_pace_go_verdict.py` smoke test.
- Next server gate: sync this branch by git bundle, run
  `/home/ajifang/miniconda3/envs/tglrec-lora/bin/python -m pytest tests/unit/test_hf_judge_equivalence.py -q`,
  wait for a clean GPU (>=43GB free), resume only `full` from `full.json.per_user.jsonl`, then run
  `scripts/cc_pace_go_verdict.py`. In practice, use chunked limits (`125,150,...,950,973`) so each
  Python process exits and releases GPU memory; the driver skips already-scored users from the
  per-user checkpoint. GO -> LoRA; KILL_OR_REFRAME -> 3-seat ARIS redesign.
