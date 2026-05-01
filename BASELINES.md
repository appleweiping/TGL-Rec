# Baseline policy

This file must be refreshed before final experiments. Do not rely on stale baseline lists.

## Baseline categories

### 1. Sanity and classic recommenders

- Popularity: catches dataset bias and evaluates whether complex models beat a trivial prior.
- item-kNN / co-occurrence: direct comparator for item transition graphs.
- BPR-MF: non-sequential collaborative filtering baseline.

### 2. Graph collaborative filtering

- LightGCN or a modern equivalent.
- Purpose: show whether temporal direction/path evidence adds value beyond static user-item graph aggregation.

### 3. Sequential recommendation

Minimum:

- GRU4Rec
- SASRec
- BERT4Rec
- TiSASRec

Add after literature refresh:

- strong contrastive/sequential models available in RecBole or official code;
- dataset-specific leaders if they are reproducible and protocol-compatible.

### 4. Time-aware / temporal models

- TiSASRec as a required baseline.
- Consider dynamic GNN baselines such as TGN-style recommenders for global-time inductive split.
- Consider survival/hazard or Hawkes-process-inspired item transition baseline if implementation cost is low.

### 5. LLM4Rec baselines

Required target:

- LLM-SRec-style method if code remains reproducible.

Optional depending on compute/API:

- P5/TallRec/ReLLa/LLaRA-style baselines;
- frozen LLM prompting with item titles and histories;
- G-Refer-style static graph translation adaptation.

## Fairness rules

- Same train/validation/test splits.
- Same item universe and filtering.
- Same candidate set for sampled ranking, or full ranking for all methods when feasible.
- Same negative sampling seed and count when using sampled ranking.
- Comparable hyperparameter search budget.
- Report whether the baseline uses item text, item IDs, graph structure, timestamps, or pretrained LLMs.

## Literature refresh checked 2026-04-29

Recommended required executable baselines:

| Baseline | Source code | License | Dataset support | Needs GPU | Needs API | Status | Notes |
|---|---|---:|---|---:|---:|---|---|
| Popularity | local or RecBole Pop | project / MIT if RecBole | all | no | no | Module 2 first | sanity baseline; full-ranking feasible |
| item-kNN | local or RecBole ItemKNN | project / MIT if RecBole | all | no | no | Module 2 first | co-occurrence comparator for TDIG |
| BPR-MF | local or RecBole BPR | project / MIT if RecBole | all implicit | no for ML-1M | no | local CPU runner added 2026-05-01 | required non-sequential CF |
| GRU4Rec | RecBole or https://github.com/hidasib/GRU4Rec | check official | all sequences | yes for real runs | no | required | use project splits, not default session splits |
| SASRec | https://github.com/kang205/SASRec or RecBole | Apache-2.0 / MIT RecBole | all sequences | yes | no | required | official code is old; RecBole easier |
| BERT4Rec | https://github.com/FeiSun/BERT4Rec or RecBole | Apache-2.0 / MIT RecBole | all sequences | yes | no | required | include replicability caution from https://arxiv.org/abs/2207.07483 |
| TiSASRec | https://github.com/JiachengLi1995/TiSASRec | no license found | timestamped sequences | yes | no | required | must-run time-aware baseline; prefer clean implementation if license blocks reuse |
| LightGCN | https://github.com/kuandeng/LightGCN or RecBole | no license found / MIT RecBole | all implicit | yes | no | required | static graph comparator; train graph only |
| LLM-SRec-style | https://arxiv.org/abs/2502.13909 / https://github.com/Sein-Kim/LLM-SRec | no license found | selected sequential datasets | yes | optional | required target after Module 2 starts | directly tests sequence-use hypothesis |
| G-Refer-style static | https://arxiv.org/abs/2502.12586 / https://github.com/Yuhan1i/G-Refer | no license found | explanation/graph rec datasets | yes | optional | optional comparator | static graph-to-language related work |
| FEARec | RecBole or https://github.com/sudaada/FEARec | MIT if RecBole | all sequences | yes | no | optional | SIGIR 2023 strong sequential baseline |
| BSARec | https://github.com/yehjin-shin/BSARec | check | ML-1M/Amazon/Yelp examples | yes | no | optional | AAAI 2024; heavier than required set |
| HSTU/generative rec | https://github.com/meta-recsys/generative-recommenders | check | repo configs | yes heavy | no | related/stretch | not needed before TDIG proof |

Module 0/1 CPU-light baseline policy:

- `popularity` and `item_knn` have a completed CPU sanity runner as of 2026-04-29:
  `tglrec evaluate sanity-baselines --dataset-dir artifacts/datasets/movielens_1m`.
  The first completed MovieLens-1M run is under `runs/20260429-ml1m-sanity-baselines-v2/`.
  It is suitable for engineering sanity checks, not paper-grade claims, because this workspace is
  not a Git checkout and the dataset artifact is not yet content-addressed with checksums.
- `bpr_mf` has a local deterministic NumPy runner as of 2026-05-01:
  `tglrec train bpr-mf --dataset-dir artifacts/datasets/movielens_1m_checksummed_20260430`.
  A reproducible sweep wrapper is also available via `tglrec train bpr-mf-sweep`, with parent-level
  `sweep_results.csv` and full child run artifacts. It is suitable for CPU engineering runs and
  small MovieLens sweeps; tune hyperparameters before citing results.
- `recbole_general` export is available as of 2026-05-01:
  `tglrec export recbole-general --dataset-dir artifacts/datasets/movielens_1m_checksummed_20260430`.
  Use it for RecBole BPR/LightGCN-style general CF baselines with project train/valid/test labels.
  Do not use this export unchanged for SASRec/BERT4Rec/TiSASRec; sequential baselines need a
  history-aware adapter.
- Add `tdig_transition_counts` as a local non-neural diagnostic baseline: directed item-item transition counts/lift from training-only events, with no language model.
- Use RecBole as an integration path where it preserves the project split files and evaluator semantics.
- Defer full SASRec, BERT4Rec, TiSASRec, LightGCN, GRU4Rec, and LLM-SRec runs until Module 2; only tiny CPU smoke runs are appropriate before GPU setup.
- Do not claim SOTA from RecBole defaults or literature numbers with incompatible splits.

## SOTA update instructions for Codex

Before main experiments:

1. Search recent WWW/SIGIR/KDD/RecSys/CIKM papers and arXiv for sequential recommendation, LLM4Rec, graph recommendation, and time-aware recommendation.
2. Prefer official repositories and papers.
3. Add only baselines that can be run fairly or explain why they cannot be run.
4. Update `configs/baselines.yaml` and this file.
5. Record exact search date and sources in `docs/literature_log.md`.
