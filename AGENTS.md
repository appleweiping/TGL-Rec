# AGENTS.md — Project instructions for Codex

## Mission

You are working on a top-tier research codebase for **Temporal Graph-to-Language Retrieval for Need-Aware Sequential Recommendation**. The goal is to produce a credible WWW/SIGIR/KDD/RecSys-level paper, not a toy implementation.

The project asks whether LLM-based recommenders truly use sequence and timestamps, or mostly follow semantic similarity. The target contribution is a diagnostic benchmark plus a temporal graph-to-language retrieval/reranking method that separates semantic similarity from next-need temporal transition.

## Working principles

- Do not ask the user for trivial confirmations. Make a reasonable default decision, document it, and keep moving.
- Ask for user intervention only when blocked by credentials, paid API budget, licensed data access, GPU/server constraints, or destructive operations.
- Prefer official sources for installation, dataset format, benchmark protocol, API behavior, and library usage. Record exact URLs and access dates in `docs/literature_log.md` or a task note.
- Never copy paper text or code into this repository unless the license permits it and attribution is preserved. Reimplement ideas cleanly.
- Do not claim SOTA unless the baselines are reproduced fairly, tuned reasonably, and evaluated with the same split/negative sampling/full-ranking protocol.
- Every experimental claim must point to a config, seed, dataset version, commit, and result file.
- Avoid premature LLM/API dependence. Build the strongest retrieval + small reranker path first; add LLM calls only when they test a clear hypothesis.

## Definition of done for code changes

A task is not done until all applicable items are true:

1. The changed code has a runnable command documented in the task note or README.
2. Unit or smoke tests pass, or the limitation is documented with the exact missing dependency.
3. The code writes machine-readable outputs under `runs/` or `artifacts/` with config and seed metadata.
4. New dataset preprocessing is deterministic and does not leak future interactions into training.
5. New evaluation code supports at least HR@K, NDCG@K, MRR@K, and candidate/full-ranking mode when feasible.
6. New model code has an ablation switch so it can be tested without the new component.
7. The change updates relevant docs: `TASKS.md`, `EXPERIMENTS.md`, or `PAPER_OUTLINE.md`.

## Research guardrails

- The main novelty is not simply adding timestamps to prompts. The novelty must include explicit diagnosis and modeling of similarity vs temporal need transition.
- Keep the causal language careful. Unless an experiment uses a valid causal design, call evidence “temporal transition evidence” or “counterfactual perturbation evidence,” not proof of causality.
- Prevent temporal leakage. When building item-item transition graphs, only use interactions available before the prediction timestamp.
- Separate candidate generation from ranking. Report both candidate recall and final ranking metrics.
- Keep a strong no-time/no-language/no-graph ablation suite.
- Include statistical reliability: multiple seeds where possible, paired bootstrap or paired t-test/Wilcoxon over users, and confidence intervals for main tables.

## Required baseline policy

At minimum, implement or integrate:

- non-sequential: popularity, item-kNN, BPR-MF or equivalent;
- graph CF: LightGCN or equivalent;
- sequential: GRU4Rec, SASRec, BERT4Rec;
- time-aware sequential: TiSASRec or a stronger verified time-aware alternative;
- recent LLM4Rec/sequential LLM baseline: LLM-SRec-style or best available reproducible code;
- G-Refer-style static graph-to-language adaptation for explanation/reranking comparison when feasible.

Before final experiments, run a literature refresh and update `BASELINES.md` with papers/code found after this repository was created.

## Data policy

- Prefer public benchmarks with timestamped user-item interactions: MovieLens-1M/20M, Amazon reviews, Steam, Yelp, and at least one larger sparse dataset if compute allows.
- Dataset scripts should download public data automatically when license permits.
- If a dataset requires manual agreement or login, write a clear `DATA_MANUAL_STEPS.md` entry and continue with alternatives.
- Use temporal leave-one-out and a global-time inductive split. Do not mix random split results into main claims.

## Experiment logging

All experiments must write:

```text
runs/<date>-<short_name>/
  config.yaml
  metrics.json
  metrics_by_segment.csv
  command.txt
  git_commit.txt
  stdout.log
  stderr.log
```

Segment-level metrics should include at least user history length, last interaction time gap, item popularity bucket, semantic-vs-transition case type, and cold/warm item status where available.

## Coding standards

- Use Python 3.10+.
- Prefer clear, typed, modular code over clever abstractions.
- Use deterministic seeds and document nondeterminism.
- Do not silently swallow errors. Fail loudly with actionable messages.
- Keep model implementations independent from dataset-specific assumptions.
- Add CLI entry points for preprocessing, graph construction, training, evaluation, and diagnostics.

## Suggested package layout

```text
src/tglrec/data/          dataset download/preprocess/splits
src/tglrec/graph/         temporal directed item graph and path retrieval
src/tglrec/text/          graph-to-language templates and item text encoders
src/tglrec/models/        rerankers, gates, dynamic GNN adapters
src/tglrec/eval/          metrics, diagnostics, statistical tests
src/tglrec/utils/         config, logging, seeds, io
```

## When running Codex subagents

Use focused roles:

- `research_worker`: implements code and runs tests.
- `reviewer`: read-only correctness, leakage, evaluation, and missing-test review.
- `lit_scout`: read-only literature and official-doc verification.
- `experiment_runner`: executes configured experiments and records results.
- `repro_auditor`: checks determinism, logs, split leakage, and fair baselines.

For complex tasks, spawn at least one reviewer in parallel before merging a large change.
