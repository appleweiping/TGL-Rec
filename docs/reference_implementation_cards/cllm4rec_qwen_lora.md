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

- Use Qwen3-8B as the shared base model only where faithful.
- Preserve the official collaborative ID-token, prompt, head, and regularization
  design.
- Score the shared candidate set under the frozen split and leakage protocol.
- Output shared prediction JSONL and candidate-adherence diagnostics.
- Evaluate only through the shared TGL-Rec evaluator.

## Implementation Status

`not_implemented`

Do not collapse this baseline into plain text reranking; the collaborative
tokens/head/regularization are the method identity.
