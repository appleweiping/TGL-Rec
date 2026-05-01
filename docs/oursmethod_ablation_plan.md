# Phase 5 Ablation Plan

This document describes non-reportable ablation interfaces for `TimeGraphEvidenceRec`. These switches are infrastructure for future experiments and are not empirical claims.

## Required Ablations

- `full`: all Phase 5 evidence interfaces enabled, no real LLM call.
- `w_o_llm`: keeps deterministic evidence scoring and disables any future LLM call.
- `w_o_retrieval`: disables evidence retrieval.
- `w_o_temporal_graph`: disables transition, time-window, and time-gap graph evidence.
- `w_o_transition_edges`: disables directed transition edges.
- `w_o_time_window_edges`: disables time-window evidence.
- `w_o_time_gap_tags`: strips gap bucket tags from evidence.
- `w_o_semantic_similarity`: disables item metadata similarity evidence.
- `w_o_user_profile`: disables recent and long-term profile blocks.
- `w_o_grounding_constraint`: disables the grounding switch for future prompt integration.
- `w_o_explanation`: disables prompt-ready evidence text in metadata.
- `encoder_only`: enables only the future encoder interface stub.
- `text_only`: keeps metadata/text evidence and disables temporal graph evidence.
- `graph_only`: keeps temporal graph evidence and disables semantic/explanation switches.

## Safety

All Phase 5 ablations are `reportable=false`. They must use the shared candidate protocol, shared prediction schema, and shared evaluator. No real API calls, LoRA training, or paper-scale experiments are part of this plan.
