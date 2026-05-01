# Research Idea: Time-Aware Graph Evidence for LLM4Rec

## Working Title

The project name is not final. Working names include:

- T-GraphRec
- TimeGraph-LLMRec
- T-Refer

Do not treat any of these names as final paper branding.

## Core Idea

This project is not a generic LLM4Rec reranking system, not a generic RAG recommender, and not a
direct G-Refer replication.

The core research question is:

> Do LLM-based recommenders truly use item order, time gaps, time windows, and user interest drift
> in interaction histories? If they do not, can a time-aware interaction graph upgrade user
> behavior evidence from static path evidence into timestamped dynamic graph evidence for LLMs or
> small models?

The target contribution is a diagnostic protocol plus a time-aware modeling mechanism. The method
should separate semantic item similarity from next-need temporal transition evidence, then test
whether timestamped graph evidence improves recommendation accuracy, explanation faithfulness,
long-tail behavior, and cold-start behavior.

## Motivation

G-Refer-style work converts graph structure into path and node text evidence so that LLMs can use
collaborative filtering graphs for explainable recommendation. Static graph paths mainly express
which entities are connected. They do not necessarily express when the connection happened, how
long the gap was, or whether two actions occurred in the same need cycle.

Lost-in-Sequence-style findings suggest that LLM4Rec systems may fail to fully exploit sequential
recommendation signals. A model can appear sequence-aware because the prompt lists a history, while
its predictions may still be driven by semantic similarity, title overlap, or popularity.

Before building OursMethod, the codebase must support diagnostic experiments that ask:

1. Does changing item order significantly change LLM recommendations?
2. Does adding timestamps or time-gap tags significantly change recommendations?
3. Are within-window relations, such as within one day, one week, or one month, stronger than
   timeless graph paths for next-item recommendation?
4. Do similarity-dominant and sequence or need-dominant transitions behave differently?
5. Does user interest drift require time-aware blocks, recurrent state, hierarchical profiles, or
   dynamic graph representations?

## Hypotheses

H1: With only item titles and history prompts, LLM recommenders are not sufficiently sensitive to
history order.

H2: Time-window item transition edges encode immediate user need better than static item
similarity edges.

H3: Translating timestamped graph evidence into text can improve LLM reranking grounding and
explanation faithfulness.

H4: A dynamic graph encoder or temporal graph encoder can support inductive representations for
new users, new items, and newly observed interactions.

H5: A strong final method should jointly use static similarity signal, sequential transition
signal, temporal-window co-occurrence signal, user interest drift signal, and LLM textual reasoning
signal.

## Key Distinction From G-Refer

G-Refer focuses on graph retrieval and graph-to-text explanation over collaborative filtering
graphs.

This project focuses on time-aware graph evidence:

- directed item transition graph;
- timestamped interaction edges;
- windowed co-occurrence edges;
- time-gap tags;
- recent versus long-term preference blocks;
- dynamic or inductive graph embeddings;
- diagnostics for whether LLMs use sequence and time information.

G-Refer can be cited as related work, but the contribution must not be "G-Refer plus timestamp."
The contribution must include a diagnostic protocol and a time-aware modeling mechanism.

## Planned Research Questions

RQ1: Do LLM-based recommenders change predictions when the same user history is reordered?

RQ2: Do LLM-based recommenders benefit from explicit timestamps or time-gap tags?

RQ3: Does a windowed directed item graph capture need-driven transitions better than
similarity-only retrieval?

RQ4: Can temporal graph evidence improve LLM reranking beyond BM25, popularity, collaborative
filtering, and static graph paths?

RQ5: Can dynamic graph encoding support inductive updates for new interactions?

RQ6: Does time-aware graph evidence improve explanation faithfulness and reduce hallucination?

## Diagnostic Experiments Before OursMethod

OursMethod must not be implemented before the infrastructure can run diagnostics and minimal
baselines. The diagnostics below are required before the final method is shaped.

### Sequence Perturbation

- original order;
- reversed order;
- shuffled order;
- sorted by popularity;
- recent-k only.

### Time Tag Ablation

- no timestamp;
- absolute timestamp;
- relative time gap;
- bucketed time gap: same day, same week, same month, old;
- recency block: short-term, mid-term, long-term.

### Graph Evidence Comparison

- static similarity edge;
- user co-consumption edge;
- directed transition edge;
- within-week transition edge;
- time-decayed edge;
- dynamic graph embedding.

### Similarity vs Need Transition

- item text similarity dominant cases;
- sequential transition dominant cases;
- repeat, complement, and substitute cases;
- category-switch cases.

### User Drift Analysis

- early history block;
- recent history block;
- hierarchical user profile;
- recurrent user profile.

## OursMethod Placeholder

Use the placeholder name `OursMethod` until the diagnostics justify a specific architecture.

OursMethod should eventually contain:

1. Time-aware graph constructor.
2. Temporal evidence retriever.
3. Time-tag graph-to-text translator.
4. Optional dynamic GNN or TGN-style encoder.
5. LLM reranker or generator with grounded temporal evidence.
6. Validity, hallucination, and explanation-faithfulness evaluator.

Do not implement OursMethod until diagnostic experiments and the Phase 2 and Phase 3 infrastructure
are ready.

## Required Infrastructure Support

The codebase must support:

- item transition graph construction;
- timestamped edge features;
- time-window edge construction;
- time-decay edge weights;
- sequence perturbation configs;
- prompt variants with and without time tags;
- static similarity versus directed transition comparison;
- per-segment metrics by time gap, user sparsity, and item popularity;
- LLM raw output storage;
- dynamic graph encoder interface.

## Non-Goals

Do not claim:

- timestamps always help;
- LLMs understand sequence by default;
- OursMethod is better before experiments;
- graph-to-text alone is the contribution;
- dynamic GNNs are necessary before diagnostics show temporal structure matters.

## Immediate Next Step

Proceed to Phase 2A: Minimal Baselines + Diagnostic Data Structures, not full OursMethod.
