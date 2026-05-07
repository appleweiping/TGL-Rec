# Reference Method Adaptation Map

## Principle

This map selects concrete senior-reference methods for faithful adaptation.

Baseline algorithm belongs to the baseline:

- preserve its training objective;
- preserve its input construction;
- preserve its scoring/reranking logic;
- preserve its special signals such as collaborative IDs, long-tail
  self-distillation, controllability alignment, or representation alignment.

Experiment protocol belongs to TGL-Rec:

- Qwen3-8B LoRA/QLoRA backbone where faithful;
- same frozen data;
- same candidate sets;
- same split;
- same metrics;
- same prediction schema.

No method below is reportable until implemented, trained, evaluated, and
diagnosed under the shared protocol.

## Selected Reference Baselines

| Baseline ID | Local reference | Method identity to preserve | Fair Qwen3-8B adaptation target |
|---|---|---|---|
| `slmrec_distill_qwen_lora` | `references/NH/11465_SLMRec_Distilling_Large_.pdf` | SLMRec-style depth/knowledge distillation for sequential recommendation | Qwen3-8B LoRA student with distillation-style supervision; preserve distillation objective rather than replacing it with generic ranking SFT |
| `llm_esr_qwen_lora` | `references/NH/NeurIPS-2024-llm-esr-large-language-models-enhancement-for-long-tailed-sequential-recommendation-Paper-Conference.pdf` | Long-tail sequential recommendation using LLM semantic embeddings, dual-view semantic/collaborative modeling, and retrieval-augmented self-distillation | Use Qwen3-derived semantic embeddings/signals with the shared candidate protocol; preserve long-tail user/item treatment and self-distillation |
| `controllable_rec_qwen_lora` | `references/NH/Aligning Large Language Models for Controllable Recommendations.pdf` | Recommendation-specific instruction tasks plus alignment for controllable recommendation and format fidelity | Use Qwen3-8B LoRA for controllable recommendation SFT/alignment tasks; preserve control labels and instruction-following objective |
| `cllm4rec_qwen_lora` | `references/NR/3589334.3645347.pdf` | Collaborative LLM recommender with user/item ID tokens, soft+hard prompting, item prediction head, and mutual regularization | Extend Qwen3-8B LoRA with collaborative ID-token/prompt structure and prediction-head scoring where feasible; do not collapse it into plain text reranking |
| `rlmrec_qwen_lora` | `references/NR/3589334.3645458.pdf` | LLM-empowered representation learning with user/item profiling and cross-view alignment between semantic and collaborative spaces | Use Qwen3-generated/user-item semantic profiles and preserve cross-view representation alignment under the shared split/candidate protocol |
| `transrec_qwen_lora` | `references/NR/3637528.3671884.pdf` | Multi-facet item identifiers and constrained/grounded generation from item-language transition paradigm | Use Qwen3-8B LoRA with multi-facet identifiers and valid-item grounding; preserve generation grounding rather than unconstrained free-form output |
| `review_pref_reasoning_qwen_lora` | `references/NR/3726302.3730055.pdf` | Review-driven personalized preference reasoning and LLM reranking | Preserve review-to-preference extraction/reasoning and reranking logic; only adapt backbone/data/candidates/evaluator |

## Source Notes

These method identities were selected from local files plus public paper pages:

- SLMRec paper page identifies the method as distilling LLMs into smaller
  sequential recommendation models, with layer redundancy and knowledge
  distillation as core ideas.
  - Public page: <https://arxiv.org/abs/2405.17890>
- LLM-ESR official NeurIPS page describes LLM semantic embeddings, dual-view
  semantic/collaborative modeling for long-tail items, and retrieval-augmented
  self-distillation for long-tail users.
  - Public page: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html>
- Controllable Recommendation paper page describes supervised
  recommendation-specific instruction tasks augmented with conventional
  recommender labels, followed by reinforcement-learning-based alignment.
  - Public page: <https://arxiv.org/abs/2403.05063>
- CLLM4Rec paper page describes adding user/item ID tokens, soft+hard prompting,
  mutual regularization, and an item prediction head.
  - Public code page: <https://github.com/yaochenzhu/LLM4Rec>
- RLMRec paper page describes LLM-empowered representation learning and
  cross-view alignment between semantic and collaborative representations.
  - Public code page: <https://github.com/HKUDS/RLMRec>
- TransRec paper page describes multi-facet identifiers and grounding generated
  identifiers to in-corpus items.
  - Public page: <https://arxiv.org/abs/2310.06491>
- Review-driven Personalized Preference Reasoning is present in the local
  reference set; public metadata confirms the title. Its exact adaptation
  requires method extraction from the local PDF before implementation.
  - Public page: <https://arxiv.org/abs/2408.06276>

## Promotion Plan

1. Extract method notes for each selected reference.
2. Create one implementation card per baseline with:
   - source paper;
   - original algorithm components;
   - preserved components;
   - adapted components;
   - unsupported components and why;
   - training command;
   - evaluation command.
3. Implement baselines in disjoint modules instead of forcing all of them into
   `sft_variants.py`.
4. Use `sft_variants.py` only for methods whose original algorithm is actually
   SFT/prompt-based.
5. Train each baseline with Qwen3-8B LoRA/QLoRA when faithful.
6. Evaluate on the same candidate protocol with shared metrics.

## Current Reportability

None of the selected reference baselines is reportable yet.

The current `reference_*_sft` variants remain scaffolds. They should be replaced
or upgraded only after the corresponding original method has been implemented
faithfully under this map.
