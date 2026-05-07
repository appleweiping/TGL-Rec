# Implementation Card: review_pref_reasoning_qwen_lora

## Source

- Local reference: `references/NR/3726302.3730055.pdf`
- Public page: <https://arxiv.org/abs/2408.06276>

## Original Method To Preserve

- Review-to-preference extraction.
- Personalized preference reasoning.
- LLM reranking based on extracted preferences.

## Shared Protocol Adaptation

- Use Qwen3-8B as the shared base model only where faithful.
- Preserve the official preference extraction, reasoning, and reranking design.
- Use shared splits/candidates/schema/evaluator.
- Preserve review-derived evidence when the dataset provides reviews.

## Implementation Status

`not_implemented`

Do not report this baseline until review-derived preference extraction and
reasoning are implemented or a no-review limitation is explicitly documented.
