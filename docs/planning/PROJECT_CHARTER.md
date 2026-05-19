# Project charter

## Working title

**Do LLM Recommenders Follow Time or Similarity? Temporal Graph-to-Language Retrieval for Sequential Recommendation**

Alternative title:

**Beyond Similarity: Time-Aware Graph Translation for LLM-based Sequential Recommendation**

## Motivation

LLM4Rec systems often serialize user histories as text, so they look sequential. But ordered text alone does not prove that the model uses order, time gaps, or next-need transitions. A user who buys a phone may next need a case, charger, or headphones, not another phone. This project treats that gap as the central scientific opportunity.

The project combines three ideas:

1. **Diagnostic first**: measure whether order, timestamps, and time-window language actually change recommendations.
2. **Temporal graph evidence**: convert timestamped user histories into directed, time-windowed transition graphs.
3. **Language bridge**: translate retrieved temporal paths into compact text so LLMs or small language encoders can consume structured CF evidence.

## Core research questions

RQ1. Do current LLM4Rec models use item order and timestamp information, or mostly semantic similarity/popularity?

RQ2. When does temporal transition evidence beat semantic similarity? Which datasets, user states, item categories, and time gaps reveal this effect?

RQ3. Can temporal graph-to-language evidence improve sequential recommendation and explanation without expensive full LLM fine-tuning?

RQ4. Can a dynamic GNN / continuous-time memory channel make the method inductive for new interactions, new users, and new items?

## Intended contributions

1. **Temporal sensitivity diagnostics** for LLM4Rec and sequential recommenders:
   - sequence shuffle sensitivity;
   - timestamp removal/randomization/adversarial perturbation;
   - semantic-similarity vs next-need candidate stress tests;
   - time-window robustness and leakage checks.

2. **Temporal Directed Item Graph (TDIG)**:
   - directed item-item transitions induced by user sequences;
   - edge features over time gaps: same session, 1 day, 1 week, 1 month, long gap;
   - transition lift, PMI, support, recency, direction asymmetry, and semantic dissimilarity.

3. **Temporal Graph-to-Language Retrieval**:
   - retrieve user-specific paths from recent history to candidates;
   - translate paths into controlled evidence sentences;
   - include both positive and negative evidence: e.g. “semantically similar but rarely follows within a week.”

4. **Need-aware gated reranker**:
   - learns when to trust semantic similarity vs temporal transition;
   - combines base sequential score, graph transition score, text evidence score, and optional dynamic GNN score.

5. **Inductive dynamic update path**:
   - optional continuous-time memory / TGN-style encoder;
   - new item representation initialized from item text/category features;
   - event updates without retraining the whole model.

## Main model sketch

For user `u`, history `H_u = [(i_1,t_1),...,(i_n,t_n)]`, and candidate `c`:

1. Candidate generation:
   - top candidates from SASRec/BERT4Rec/TiSASRec or another base sequential model;
   - top transition candidates from TDIG;
   - semantic nearest neighbors from item text embeddings;
   - popularity and hard negatives for diagnostics.

2. Temporal graph retrieval:
   - direct transitions from recent items to candidate;
   - 2-hop paths through bridging items;
   - window-conditioned transitions: same session, within 1 day, within 1 week, within 1 month;
   - user-segment-conditioned evidence when enough support exists.

3. Evidence translation:
   - deterministic templates, not free-form hallucinated explanation;
   - evidence is short, structured, and auditable;
   - every sentence maps back to graph statistics.

4. Reranking:

```text
score(u,c) = base_seq_score(u,c)
           + gate(u,c) * transition_score(u,c)
           + (1 - gate(u,c)) * semantic_score(u,c)
           + text_evidence_score(u,c)
           + optional_dynamic_gnn_score(u,c)
```

The gate is trained from user state and candidate evidence features, including recency, time gap, history entropy, category switch likelihood, transition support, semantic similarity, and direction asymmetry.

## Why this can be stronger than a simple timestamp prompt

A timestamp prompt only says “A happened 3 days before B.” The proposed method estimates whether that time gap is statistically meaningful across users, whether the edge is directional, whether the candidate is a complementary next need or just a similar item, and whether the current user state supports a short-term transition or long-term preference.

## Risk plan

Risk: timestamps do not help on MovieLens.
Mitigation: include Amazon/Steam/Yelp and segment analyses; entertainment ratings may be less need-transition-heavy than e-commerce.

Risk: LLM channel adds cost but no ranking gain.
Mitigation: keep LLM optional; main ranking can be graph retrieval + small reranker; use LLM primarily for explanation and qualitative evidence.

Risk: dynamic GNN is hard to tune.
Mitigation: treat dynamic GNN as a stage-2 channel; first prove TDIG + gate works; use TGN/EvolveGCN as ablation or inductive setting.

Risk: baseline wins disappear under full ranking.
Mitigation: report both sampled and full ranking where feasible; if full ranking is expensive, document sampled protocol and include candidate recall.

Risk: “SOTA” target is too broad.
Mitigation: define SOTA per dataset/protocol and reproduce the strongest open baselines. Do not compare against numbers from incompatible splits.
