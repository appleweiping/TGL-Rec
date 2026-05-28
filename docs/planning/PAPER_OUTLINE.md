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

Datasets (4 domains, same-candidate protocol from Pony):

- Amazon Beauty (973 users, 101 candidates each)
- Amazon Books (10,000 users, 101 candidates each)
- Amazon Electronics (10,000 users, 101 candidates each)
- Amazon Movies (10,000 users, 101 candidates each)

Official Baselines (8, from Pony/Uncertainty shared protocol):

- LLM2Rec (LLM-based collaborative filtering)
- LLM-ESR (LLM enhancement for sequential recommendation)
- LLMEmb (LLM embedding for recommendation)
- RLMRec (representation learning meets LLM recommendation)
- IRLLRec (intent-aware reinforcement learning LLM recommendation)
- ELMRec (efficient LLM recommendation)
- ProEx (profile-based explanation recommendation)
- ProMax (profile maximization recommendation)

All baselines use Qwen3-8B backbone, same splits, same candidates, same metrics.

Metrics:

- MRR, HR@5, HR@10, NDCG@5, NDCG@10
- Parse success rate, candidate adherence
- Paired statistical tests (McNemar, Wilcoxon signed-rank)

Ablations (7 variants):

- No temporal graph (alpha forced to 0)
- No need-gate (fixed alpha=0.5)
- No semantic trap penalty
- No time-window edges
- No LLM reranking (gate scores only)
- No recency signal
- No contrastive evidence

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
