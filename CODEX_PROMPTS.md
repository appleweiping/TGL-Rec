# Codex prompts

Use these prompts from the repository root. They are designed for Codex automation and subagent workflows.

## Prompt 1 — Initial repository build-out

```text
Read AGENTS.md, PROJECT_CHARTER.md, ROADMAP.md, EXPERIMENTS.md, BASELINES.md, and configs/*.yaml.

Goal: turn this scaffold into a runnable research codebase skeleton without doing toy shortcuts.

Tasks:
1. Create the package layout under src/tglrec/ exactly as AGENTS.md suggests.
2. Add config loading, deterministic seed utilities, logging utilities, and a CLI stub.
3. Add dataset preprocessing interfaces and implement MovieLens-1M preprocessing if public download works; otherwise write a clear manual step and implement local-file ingestion.
4. Add a small synthetic dataset only for unit tests, not as a research result.
5. Add tests for temporal ordering, no future leakage, ID mapping stability, metric correctness, and config parsing.
6. Update TASKS.md with completed work and next tasks.

Constraints:
- Do not ask me trivial questions. Choose sensible defaults.
- Use official docs for dataset URLs and library behavior.
- Do not install huge GPU dependencies unless needed; document optional installs.
- All new commands must be documented in README.md.

Done when:
- pytest passes for the implemented parts.
- A user can run one command to preprocess a local or downloaded MovieLens-1M file.
- runs/ or artifacts/ output format is defined.
```

## Prompt 2 — Literature scout

```text
Act as the lit_scout subagent if available; otherwise act read-only.

Goal: refresh the baseline and related-work list for a 2026 top-tier recommendation paper.

Read BASELINES.md and docs/literature_log.md if it exists. Search official paper pages, arXiv, OpenReview/ACM pages, and official GitHub repositories for:
- LLM4Rec sequential recommendation after 2024;
- G-Refer and graph-to-language recommendation;
- Lost in Sequence / LLM-SRec and follow-ups;
- time-aware sequential recommendation beyond TiSASRec;
- dynamic GNN / temporal graph recommenders;
- strong RecBole-compatible sequential baselines.

Write or update docs/literature_log.md with:
- citation, venue/date, core idea, code URL, license if visible;
- whether it is a must-run baseline, optional baseline, or related work only;
- estimated compute/API needs;
- risks in reproducing it.

Update BASELINES.md and configs/baselines.yaml accordingly.

Do not copy abstracts verbatim. Summarize in your own words and include links.
```

## Prompt 3 — Diagnostic benchmark implementation

```text
Goal: implement the diagnostic benchmark described in EXPERIMENTS.md Module D1-D5.

Use the current data pipeline and evaluator. Implement:
1. history shuffle perturbation;
2. order reversal perturbation;
3. timestamp removal/randomization/window-swap perturbation;
4. similarity-vs-transition candidate stress-test generator;
5. metric outputs for Sequence Sensitivity Index, Time Sensitivity Index, Semantic Trap Rate, Transition Win Rate, and Need Switch Accuracy.

Use synthetic tests to verify each perturbation. Then run the diagnostics on the smallest processed real dataset available.

Spawn reviewer and repro_auditor if available before finalizing.

Done when:
- tests pass;
- diagnostic CLI command is documented;
- output CSV/JSON schema is stable;
- a sample diagnostic report is written under artifacts/diagnostics/ or runs/.
```

## Prompt 4 — Temporal Directed Item Graph implementation

```text
Goal: implement TDIG construction and path retrieval.

Implement:
1. directed item-item transitions from timestamped user histories;
2. time-gap buckets: same_session, within_1d, within_1w, within_1m, long_gap;
3. edge stats: support, decayed support, transition probability, lift/PMI, direction asymmetry, recency, gap histogram;
4. semantic similarity placeholder interface using item text embeddings when available;
5. leakage-safe graph building from training split only;
6. retrieval APIs for direct and 2-hop paths from recent user items to a candidate.

Add tests for directionality, bucket assignment, no leakage, and retrieval determinism.

Done when:
- TDIG can be built from a processed dataset;
- top transition candidates can be exported;
- path evidence has stable IDs and source stats.
```

## Prompt 5 — Temporal graph-to-language evidence

```text
Goal: implement faithful graph-to-language translation for TDIG paths.

Implement deterministic templates that produce compact evidence sentences for:
- direct transition evidence;
- 2-hop path evidence;
- repeated recent transition evidence;
- negative evidence: semantically similar but low transition support;
- temporal contrast: strong within-week but weak long-gap, or vice versa.

Every generated sentence must carry metadata linking back to edge/path IDs and numeric stats. Add token budget controls.

Do not use an LLM to invent evidence. Use templates first.

Done when:
- tests prove all claims come from graph stats;
- examples are exported for at least 20 user-candidate cases;
- the evidence can be consumed as text and as structured JSON.
```

## Prompt 6 — Need-aware reranker

```text
Goal: implement and evaluate the need-aware gated reranker.

Build feature extraction for user-candidate pairs:
- base sequential model score if available;
- TDIG transition/path scores;
- semantic similarity score;
- item popularity and candidate source indicators;
- user drift features: recency gaps, history length, category entropy, repeated-intent flags;
- text-evidence embedding feature if available.

Train a pairwise or listwise reranker. Implement gate g(u,c) that interpolates transition and semantic evidence.

Run ablations:
- no graph;
- graph no time;
- graph with time;
- no gate;
- gate;
- no language evidence;
- language evidence.

Done when:
- reranker improves over at least one strong base model on a real dataset or produces a clear failure report;
- ablations and diagnostics are logged;
- reviewer and repro_auditor subagents have checked leakage and evaluation fairness.
```

## Prompt 7 — Review current branch

```text
Review this branch against main. Spawn subagents if available:
- reviewer: find correctness, leakage, evaluation, and missing-test issues;
- repro_auditor: check determinism, output logging, split integrity, and fair baseline usage;
- lit_scout: verify any external-library or paper claims added in this branch.

Summarize findings by severity:
1. must fix before experiments;
2. should fix before paper claims;
3. nice to improve.

Do not focus on style unless it hides a real risk.
```

## Prompt 8 — Experiment runner

```text
Act as experiment_runner if available.

Goal: run the next experiment block from configs/experiment_matrix.yaml.

Steps:
1. Verify environment and data availability.
2. Run the smallest smoke test first.
3. Run the configured real experiment.
4. Save command, config, git commit, stdout/stderr, metrics, and segment metrics under runs/.
5. Update a markdown summary with results, anomalies, and next recommended action.

If blocked by missing GPU, API key, or manual dataset access, write the exact blocker and continue with a CPU-feasible experiment.
```

## Prompt 9 — Paper-writing pass

```text
Read all result summaries, EXPERIMENTS.md, PROJECT_CHARTER.md, and PAPER_OUTLINE.md.

Goal: update PAPER_OUTLINE.md into a paper-ready draft skeleton.

Do not invent results. Use placeholders where results are missing. For each claim, cite a run directory, table, or figure ID. Flag unsupported claims explicitly.

Include:
- problem framing;
- diagnostic findings;
- method;
- experiments;
- ablations;
- limitations;
- reproducibility checklist.
```
