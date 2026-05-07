# Implementation Card: transrec_qwen_lora

## Source

- Local reference: `references/NR/3637528.3671884.pdf`
- Public page: <https://arxiv.org/abs/2310.06491>

## Original Method To Preserve

- Multi-facet item identifiers.
- Transition from generated identifiers to in-corpus items.
- Grounded generation constraints.

## Shared Protocol Adaptation

- Use a Qwen3-8B-compatible generation path only where faithful.
- Preserve the official identifier and grounding design.
- Ground outputs to the shared candidate set.
- Report validity and hallucination diagnostics.

## Implementation Status

`not_implemented`

Do not replace this with unconstrained free-form generation; identifier grounding
is the method identity.
