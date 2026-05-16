# Pony Official Baseline Provenance Map

## Principle

TGL-Rec uses the Pony/Uncertainty official same-candidate baseline suite as the
active main comparison set. The project should not spend new budget rebuilding a
different senior-reference baseline queue unless the user explicitly changes the
paper strategy.

Experiment protocol belongs to TGL-Rec and Pony jointly:

- same four-domain task families;
- same event rows and candidate sets;
- same score schema: `source_event_id,user_id,item_id,score`;
- same Qwen3-8B declared-adaptation policy where an official baseline consumes
  LLM/text representations;
- same shared evaluator and paired event IDs.

Method identity belongs to each official baseline:

- preserve official architecture, losses, representation modules, graph/intent
  or profile components, and scoring heads;
- use official/default or recommended baseline hyperparameters;
- do not collapse a baseline into generic prompt SFT.

## Active Map

| Baseline | Method identity | Official repo | Status |
|---|---|---|---|
| `llm2rec` | LLM2Rec Qwen3-8B embedding plus SASRec path | `https://github.com/HappyPointer/LLM2Rec` | completed |
| `llmesr` | LLM-ESR semantic/collaborative long-tail sequential path | `https://github.com/Applied-Machine-Learning-Lab/LLM-ESR` | completed |
| `llmemb` | LLMEmb Qwen3-8B embedding alignment plus sequential recommender | `https://github.com/Applied-Machine-Learning-Lab/LLMEmb` | completed |
| `rlmrec` | RLMRec graph contrastive and semantic alignment path | `https://github.com/HKUDS/RLMRec` | completed |
| `irllrec` | IRLLRec intent representation learning path | `https://github.com/wangyu0627/IRLLRec` | completed |
| `elmrec` | ELMRec high-order graph interaction path | `https://github.com/WangXFng/ELMRec` | completed |
| `proex` | 2026 ProEx profile-enhanced recommendation path | `https://github.com/BlueGhostYi/ProRec` | completed |
| `promax` | 2026 ProMax profile-enhanced recommendation path | `https://github.com/BlueGhostYi/ProRec` | pending |
| `setrec` | SETRec semantic identifier path | `https://github.com/Linxyhaha/SETRec` | blocked/replaced |

Pinned commits, domain status, summary paths, and evidence archive paths are
machine-readable in `configs/baselines/pony_official_external.yaml`.

## Promotion And Import Rules

A Pony baseline row can enter a completed TGL-Rec main table only after:

1. the manifest marks it completed for the declared domains;
2. provenance records official repo and pinned commit;
3. score rows use `source_event_id,user_id,item_id,score`;
4. score keys exactly match candidate rows, with no missing, extra, duplicate,
   blank, or non-finite scores;
5. imported metrics are produced by the shared evaluator;
6. paired event IDs remain available for significance tests.

`promax` remains planned but pending until it satisfies the same criteria across
the declared domains. `setrec` remains blocked/replaced unless the user
explicitly reopens it.

## Historical Appendix

The previous SLMRec/CLLM4Rec/TransRec/review-preference adaptation queue is
historical. Those papers can still inform related work and reviewer pressure
tests, but the active baseline execution path is Pony official baseline reuse
and migration.
