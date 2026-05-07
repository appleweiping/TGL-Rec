# Implementation Card: slmrec_distill_qwen_lora

## Source

- Local reference: `references/NH/11465_SLMRec_Distilling_Large_.pdf`
- Public page: <https://arxiv.org/abs/2405.17890>

## Original Method To Preserve

- Distillation from a larger recommendation-capable LLM into a smaller
  recommender.
- Teacher-student supervision and distillation loss.
- Sequential recommendation training signal.

## Shared Protocol Adaptation

- Use Qwen3-8B LoRA/QLoRA as the controlled backbone where the original method
  uses an LLM.
- Use TGL-Rec frozen splits and candidates.
- Output shared prediction JSONL.

## Implementation Status

`not_implemented`

Do not train or report this baseline until the distillation objective and
teacher/student data path are implemented.
