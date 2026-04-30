# Roadmap

This roadmap is designed to prevent the project from getting stuck at toy-MVP level. Each module has a research purpose, engineering output, and acceptance criteria.

## Module 0 — Literature and benchmark lock

Goal: establish current baselines and fair protocols before writing main model code.

Tasks:

- Read and summarize G-Refer, Lost in Sequence / LLM-SRec, TiSASRec, SASRec, BERT4Rec, LightGCN, TGN, EvolveGCN, and recent LLM4Rec papers.
- Identify reproducible codebases and licenses.
- Decide final dataset list and split protocol.
- Update `BASELINES.md` with exact papers, repositories, and command plans.

Acceptance criteria:

- `docs/literature_log.md` contains paper claims, links, code availability, license notes, and relevance.
- `configs/datasets.yaml` and `configs/baselines.yaml` are updated.
- At least one non-LLM, one time-aware, one graph, and one LLM4Rec baseline are executable or have a documented integration path.

## Module 1 — Data pipeline and leakage-safe splits

Goal: create reliable timestamped sequential recommendation data.

Tasks:

- Implement dataset download where license permits.
- Normalize user, item, timestamp, rating/event fields.
- Filter by minimum user/item interactions.
- Produce two split types:
  - per-user leave-one-out by time;
  - global-time inductive split.
- Save metadata and item text fields for language/semantic encoders.

Acceptance criteria:

- One command produces processed MovieLens-1M.
- One command produces at least one Amazon/Steam/Yelp dataset.
- Unit tests confirm ordering, no future leakage, stable IDs, and deterministic output.

## Module 2 — Baseline reproduction

Goal: avoid weak-baseline papers.

Tasks:

- Implement or integrate popularity, item-kNN, BPR-MF, LightGCN.
- Implement or integrate GRU4Rec, SASRec, BERT4Rec, TiSASRec.
- Integrate strongest feasible LLM4Rec baseline, likely LLM-SRec-style or an updated reproducible alternative after literature refresh.
- Use shared evaluator and shared splits.

Acceptance criteria:

- Baseline metrics are saved under `runs/` with configs and seeds.
- At least three seeds for main small datasets.
- Results are comparable with reported ranges or discrepancies are explained.

## Module 3 — Diagnostic benchmark

Goal: prove the problem exists before claiming a solution.

Diagnostics:

1. **History shuffle**: shuffle user histories while preserving item multiset.
2. **Order reversal**: reverse recent sequence and evaluate performance drop.
3. **Timestamp removal**: keep order but remove time gaps.
4. **Timestamp randomization**: preserve item order but randomize time gaps within user.
5. **Window perturbation**: swap within-week tags with long-gap tags.
6. **Similarity-vs-transition stress test**:
   - semantic candidate: nearest text-embedding neighbor to recent item;
   - transition candidate: high-lift next item under time window;
   - true target and hard negatives.

Acceptance criteria:

- `diagnostics_report.md` and CSV/JSON outputs exist.
- Metrics include NDCG/HR/MRR deltas and confidence intervals.
- At least one figure shows where LLM/sequential models fail or ignore time.

## Module 4 — Temporal Directed Item Graph (TDIG)

Goal: build the core time-aware retrieval structure.

Tasks:

- Construct directed edges from consecutive and bounded-window transitions.
- Store per-edge time-gap histogram and decayed statistics.
- Compute transition lift/PMI, support, recency, direction asymmetry, semantic similarity, and complementarity score.
- Implement leakage-safe graph construction for each prediction timestamp or split.

Acceptance criteria:

- Graph construction is deterministic and tested.
- Retrieval returns auditable paths with source statistics.
- TDIG candidate recall is measured independently of final reranking.

## Module 5 — Temporal graph-to-language translation

Goal: convert graph paths into compact, faithful text evidence.

Tasks:

- Define deterministic templates for direct and multi-hop paths.
- Generate positive and negative evidence.
- Enforce token budget and evidence diversity.
- Map each sentence back to graph edge IDs and stats.

Acceptance criteria:

- Evidence generation has tests.
- No hallucinated claims: every number and relation comes from graph stats.
- Evidence examples are saved for manual inspection.

## Module 6 — Need-aware gated reranker

Goal: beat strong baselines by learning when to trust semantic similarity vs temporal transition.

Tasks:

- Build feature table for `(user, candidate)` pairs.
- Include base sequential score, semantic score, transition score, path score, recency, user drift, item popularity, and optional text evidence embedding.
- Train pairwise or listwise reranker.
- Train gate `g(u,c)` that interpolates transition and similarity evidence.

Acceptance criteria:

- Reranker improves over strongest reproduced base model on at least one dataset without LLM API calls.
- Ablations show which component contributes.
- Similarity-vs-transition stress test improves meaningfully.

## Module 7 — Optional dynamic GNN / inductive update

Goal: handle new interactions and new items more naturally.

Tasks:

- Implement TGN-style continuous-time memory or integrate a vetted dynamic graph package.
- Initialize new item embeddings from text/category features.
- Use dynamic score as an additional reranker feature.
- Evaluate global-time split, cold/warm item segmentation, and streaming updates.

Acceptance criteria:

- Inductive evaluation protocol is documented.
- Dynamic module improves cold/new interaction metrics or is honestly reported as non-helpful.
- Runtime and memory overhead are measured.

## Module 8 — Paper-quality analysis

Goal: produce a complete, defensible paper package.

Tasks:

- Main tables across datasets and metrics.
- Ablation table: no time, no graph, no language, no gate, no dynamic GNN, no negative evidence.
- Diagnostic plots and case studies.
- Efficiency table: training time, inference latency, memory.
- Error analysis by time gap, category switch, popularity, user history length.

Acceptance criteria:

- `PAPER_OUTLINE.md` is filled with results and figure/table IDs.
- Reproducibility checklist is complete.
- Reviewer-style critique is addressed before submission.
