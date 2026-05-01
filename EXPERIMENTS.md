# Experiment protocol

## Stage gates

The build order is governed by `configs/stage_gates.yaml` and summarized in
`docs/experiment_gates.md`.

- Local CPU/no API first: data manifests, leakage checks, sanity baselines, BPR-MF, TDIG recall,
  semantic-vs-transition stress sets, and RecBole general-CF export.
- GPU/server/no API second: LightGCN, GRU4Rec, SASRec, BERT4Rec, TiSASRec or a verified time-aware
  alternative, all under the project splits.
- Small reranker/no API third: TDIG feature table and need-aware gate with no-time/no-graph/no-text
  ablations.
- Local LLM/LoRA fourth: only for LLM-SRec-style reproduction or controlled graph-to-language
  evidence ablations after non-LLM baselines are strong.
- Hosted API last: optional diagnostic/explanation ablation after budget, data policy, prompt
  cache, and request logging are approved.

## Datasets

Initial:

- MovieLens-1M: quick iteration, stable timestamped benchmark.
- Amazon review category dataset: e-commerce need transitions, stronger test of complementarity.

Expansion:

- MovieLens-20M if compute permits.
- Steam: gaming sequence and temporal interest drift.
- Yelp: local/business temporal patterns.
- Larger Amazon categories for final scalability tests.

## Splits

### Split A: temporal leave-one-out per user

For each user ordered by timestamp:

- train: all but last two interactions;
- validation: second last interaction;
- test: last interaction.

Use only training interactions to build graphs, item popularity, and training negatives.

### Split B: global-time inductive split

Pick global timestamp cutoffs:

- train: events before `T_train`;
- validation: events in `[T_train, T_val)`;
- test: events after `T_val`.

Use this to evaluate new interactions, new users/items, and streaming updates.

## Metrics

Ranking:

- HR@5/10/20
- NDCG@5/10/20
- MRR@10/20
- candidate recall@K before reranking

Diagnostics:

- Sequence Sensitivity Index: relative metric drop after history shuffle or reversal.
- Time Sensitivity Index: relative metric drop after timestamp removal/randomization/adversarial time tags.
- Semantic Trap Rate: frequency that the model ranks a semantically similar hard negative above the true next item.
- Transition Win Rate: frequency that the model ranks high-lift temporal transition candidate above semantic neighbor when the true next item is transition-like.
- Need Switch Accuracy: accuracy on cases where target category differs from recent item category but follows strong temporal transition evidence.

Reliability:

- multiple seeds for stochastic models;
- paired bootstrap over users for confidence intervals;
- report p-values or confidence intervals for main comparisons.

## Diagnostic experiments

### D1. History shuffle

Question: does model use sequence order?

Procedure:

- Keep the same item multiset in each user history.
- Shuffle all history positions except the target.
- Evaluate original vs shuffled.
- For LLM prompts, keep text length identical.

Expected finding:

- A truly sequential model should degrade.
- If an LLM4Rec method barely changes, it is likely relying on item bag, semantics, or popularity.

CPU sanity command:

```bash
py -3.12 -m tglrec.cli evaluate history-perturbations --dataset-dir artifacts/datasets/movielens_1m --output-dir runs/20260430-ml1m-history-perturbations-v2 --ks 5 10 20 --item-knn-max-history-items 20 --cooccurrence-history-window 20 --seed 2026
```

This command writes `metrics.json`, `metrics_by_perturbation.csv`, `metrics_delta.csv`,
`metrics_by_case.csv`, and `metrics_by_segment.csv`. `metrics_by_case.csv` records one row per
case, model, and non-original perturbation with the target rank under the original and perturbed
histories plus hit deltas for each configured cutoff. For this initial CPU sanity slice, history
perturbations fix the model-input item window first and then alter only the per-user scoring
history order passed to popularity/item-kNN; global training statistics remain timestamp-strict and
unchanged. Popularity and the current item-kNN scorer are expected negative controls: they should be
invariant to order perturbations because they do not model sequence direction after the input item
multiset is fixed. Use this artifact only to validate the diagnostic pipeline shape, not as
evidence that sequence models use or ignore order. The pre-correction run
`runs/20260430-ml1m-history-perturbations/` is obsolete.

### D2. Order reversal

Question: does the model distinguish past-to-future direction?

Procedure:

- Reverse recent L interactions.
- Keep timestamps either reversed consistently or removed depending on variant.
- Evaluate drop.

### D3. Timestamp ablation and perturbation

Variants:

- no time tag;
- absolute timestamp;
- relative time gap;
- bucket tag: same session, 1 day, 1 week, 1 month, long gap;
- randomized time bucket;
- adversarial bucket swap.

Question: do time tags change model behavior beyond adding noise?

### D4. Similarity vs transition stress test

For each test case, build a small candidate set:

- true target;
- semantic hard negative: high text similarity to recent item but low transition lift;
- transition hard candidate: high transition lift under the relevant time bucket;
- popularity hard negative;
- random negatives.

Report how often each model follows semantic similarity instead of transition evidence.

### D5. Within-week effect

Question: does “within a week” materially strengthen item-item connection?

Procedure:

- For edge `(A,B)`, compute `P(B next | A, time_bucket)` and lift by bucket.
- Compare within-week vs long-gap edges.
- Train/evaluate TDIG variants with no time, binary within-week tag, multi-bucket tag, and continuous log time gap.

## Main method ablations

- Base sequential model only.
- + TDIG candidate generation.
- + TDIG path features.
- + temporal graph-to-language evidence text embedding.
- + need-aware gate.
- + negative evidence.
- + dynamic GNN score.
- Full model.

Time ablations:

- no timestamp;
- absolute timestamp;
- relative time gap;
- bucketed time tag;
- learned time decay;
- edge histogram over time buckets.

Graph ablations:

- undirected graph;
- directed graph;
- static directed graph without time features;
- temporal directed graph;
- user-segment-conditioned temporal graph.

Language ablations:

- no language evidence;
- template text only;
- text embedding only;
- frozen LLM reranker on top-K;
- distilled small reranker.

## Main baselines

Classic:

- Popularity
- item-kNN / co-occurrence
- BPR-MF
- LightGCN

Sequential:

- GRU4Rec
- SASRec
- BERT4Rec
- TiSASRec
- recent contrastive/sequential baselines after literature refresh

Graph/session:

- SR-GNN or modern session graph baseline if session datasets are used
- dynamic/TGN-style recommender if feasible

LLM4Rec:

- LLM-SRec-style baseline or stronger updated open-source sequential LLM baseline
- ReLLa/LLaRA/P5/TallRec-style methods if reproducible under the same protocol
- G-Refer-style static graph-to-language adaptation for explanation/reranking comparison

## Acceptance bar for final claim

The paper should only claim improvement over SOTA if:

1. the strongest baseline is reproduced or fairly re-run under the same split;
2. hyperparameter tuning budgets are documented and comparable;
3. improvement holds on more than one dataset or is framed as domain-specific;
4. statistical uncertainty is reported;
5. ablations show the proposed temporal graph-to-language/gating components are responsible.
