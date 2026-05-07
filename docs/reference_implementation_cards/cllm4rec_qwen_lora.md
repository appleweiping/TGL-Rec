# Implementation Card: cllm4rec_qwen_lora

## Source

- Local reference: `references/NR/3589334.3645347.pdf`
- Public code page: <https://github.com/yaochenzhu/LLM4Rec>

## Original Method To Preserve

- User and item ID tokens.
- Soft and hard prompting.
- Item prediction head.
- Mutual regularization between text and collaborative signals.

## Shared Protocol Adaptation

- Attach collaborative ID/prompt structure to Qwen3-8B LoRA where feasible.
- Score the shared candidate set.
- Output shared prediction JSONL and candidate-adherence diagnostics.

## Implementation Status

`not_implemented`

Do not collapse this baseline into plain text reranking; the collaborative
tokens/head/regularization are the method identity.
