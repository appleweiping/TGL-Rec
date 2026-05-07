# Implementation Card: llm_esr_qwen_lora

## Source

- Local reference:
  `references/NH/NeurIPS-2024-llm-esr-large-language-models-enhancement-for-long-tailed-sequential-recommendation-Paper-Conference.pdf`
- Public page:
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html>

## Original Method To Preserve

- LLM semantic signals for long-tailed sequential recommendation.
- Dual-view semantic and collaborative modeling.
- Retrieval-augmented self-distillation.
- Explicit long-tail user/item treatment.

## Shared Protocol Adaptation

- Use Qwen3-derived semantic signals where faithful.
- Preserve long-tail split/diagnostics under the shared candidate protocol.
- Use the frozen shared split, candidate set, prediction schema, and evaluator.
- Output shared prediction JSONL.

## Implementation Status

`not_implemented`

Do not reduce this baseline to a generic long-tail prompt. The self-distillation
and dual-view logic must be implemented before training/reporting.
