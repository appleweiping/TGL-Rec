# Paper outline

## Title

Do LLM Recommenders Follow Time or Similarity? Temporal Graph-to-Language Retrieval for Sequential Recommendation

## Abstract

Placeholder. Must include diagnostic finding, method, datasets, and main quantitative results after experiments exist.

## 1. Introduction

Core story:

- LLM4Rec systems are often evaluated as sequential recommenders, but sequence serialization does not guarantee sequence understanding.
- Real user behavior often reflects next-need transitions rather than semantic similarity.
- Time gaps matter: A -> B within three days may mean something different from A -> B after six months.
- We diagnose the problem and propose temporal graph-to-language retrieval to provide faithful, time-aware CF evidence.

Claims to support later:

- Existing models show limited sequence/time sensitivity under perturbations.
- Similarity hard negatives expose failures masked by standard sampled evaluation.
- Temporal graph-to-language evidence improves ranking and explanation.

## 2. Related work

Sections:

- Sequential recommendation.
- Time-aware sequential recommendation.
- Graph neural recommendation and dynamic graphs.
- LLM4Rec and graph-to-language recommendation.
- Diagnostic/evaluation robustness for recommenders.

## 3. Problem formulation

Define:

- timestamped user-item sequence;
- next-item prediction;
- semantic similarity candidate;
- temporal transition candidate;
- temporal directed item graph;
- evaluation protocol.

## 4. Diagnostic framework

Diagnostics:

- history shuffle and reversal;
- timestamp removal/randomization/window swap;
- similarity-vs-transition stress test;
- within-week edge strength analysis;
- user interest drift segmentation.

## 5. Method: TGLRec

Components:

1. Temporal Directed Item Graph construction.
2. Temporal path retrieval.
3. Graph-to-language evidence translation.
4. Need-aware gated reranker.
5. Optional dynamic GNN inductive channel.

## 6. Experiments

Datasets:

- MovieLens-1M/20M.
- Amazon category datasets.
- Steam/Yelp if included.

Baselines:

- popularity, item-kNN, BPR-MF;
- LightGCN;
- GRU4Rec, SASRec, BERT4Rec, TiSASRec;
- LLM-SRec-style or updated LLM4Rec baselines;
- G-Refer-style static graph-to-language comparator.

Metrics:

- HR/NDCG/MRR@K;
- diagnostic metrics;
- efficiency;
- explanation faithfulness if explanation is evaluated.

## 7. Results

Tables to fill:

- Main ranking table.
- Diagnostic sensitivity table.
- Similarity-vs-transition stress table.
- Ablation table.
- Inductive/global-time split table.
- Efficiency table.

Figures to fill:

- Sequence/time sensitivity plot.
- Edge strength by time bucket.
- Gate behavior by user state and time gap.
- Case study evidence paths.

## 8. Analysis

Questions:

- When does time help?
- When does similarity dominate?
- Does language evidence help ranking or mainly explanation?
- Does dynamic GNN help new interactions/items?
- What are failure cases?

## 9. Limitations

Potential limitations:

- Dataset dependency of need-transition effects.
- LLM inference cost if used.
- Template language may be less expressive than free-form LLM evidence.
- Time tags can encode spurious seasonal/popularity effects if not controlled.

## 10. Reproducibility

Include:

- code and data processing details;
- seeds;
- hardware;
- hyperparameter search;
- licenses;
- full configs.
