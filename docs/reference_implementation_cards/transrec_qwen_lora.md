# Implementation Card: transrec_qwen_lora

## Source

- Local reference: `references/NR/3637528.3671884.pdf`
- Public page: <https://arxiv.org/abs/2310.06491>

## Original Method To Preserve

- Multi-facet item identifiers.
- Transition from generated identifiers to in-corpus items.
- Grounded generation constraints.

## Shared Protocol Adaptation

- Use Qwen3-8B LoRA for identifier generation if faithful.
- Ground outputs to the shared candidate set.
- Report validity and hallucination diagnostics.

## Implementation Status

`not_implemented`

Do not replace this with unconstrained free-form generation; identifier grounding
is the method identity.
