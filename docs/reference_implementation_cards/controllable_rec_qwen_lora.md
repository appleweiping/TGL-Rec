# Implementation Card: controllable_rec_qwen_lora

## Source

- Local reference:
  `references/NH/Aligning Large Language Models for Controllable Recommendations.pdf`
- Public page: <https://arxiv.org/abs/2403.05063>

## Original Method To Preserve

- Recommendation-specific instruction tasks.
- Control labels or controllability constraints.
- Alignment objective for controllable recommendation.
- Format fidelity and candidate grounding constraints.

## Shared Protocol Adaptation

- Use Qwen3-8B as the shared base model only where faithful.
- Preserve the official instruction/control adaptation and alignment design.
- Use shared splits, candidates, prediction schema, and evaluator.
- Preserve control-condition evaluation when the data provides control labels.

## Implementation Status

`not_implemented`

Do not report this baseline until the controllability labels/objective are
faithfully adapted.
