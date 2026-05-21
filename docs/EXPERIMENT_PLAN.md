# TGL-Rec Experiment Plan (ARIS Format)

## Target Venue
Top-conference level: KDD / WWW / RecSys / SIGIR 2026-2027

## Research Question
Can explicit temporal graph evidence with a learned need-gate improve LLM-based sequential recommendation over methods that either ignore temporal signals or handle them implicitly?

## Hypothesis (Falsifiable)
H1: TGL-Rec with learned need-gate achieves statistically significant improvement (p<0.05, paired t-test) over the best Pony official baseline on at least 3 of 4 domains in MRR and NDCG@10.

H2: The need-gate activates more strongly (higher α) for users with high transition pressure than for stable-preference users, and this activation correlates with improved ranking quality.

H3: Removing any single core component (TDIG, need-gate, evidence text, LoRA) causes statistically significant degradation, proving each is necessary.

## Experiment Blocks

### Block 1: Observation (Pain Point Validation)
**Goal**: Prove that base Qwen3-8B underuses temporal signals.

| Experiment | Method | Domains | Seeds | Evidence Level |
|-----------|--------|---------|-------|----------------|
| obs-base | Base Qwen3-8B (no adapter) | 4 | 1 | diagnostic |
| obs-shuffle | Base Qwen3-8B, shuffled history | 4 | 1 | diagnostic |
| obs-reverse | Base Qwen3-8B, reversed history | 4 | 1 | diagnostic |
| obs-recent-only | Base Qwen3-8B, last 3 items only | 4 | 1 | diagnostic |
| obs-no-time | Base Qwen3-8B, timestamps removed | 4 | 1 | diagnostic |

**Success criterion**: Shuffling/reversing history does NOT significantly hurt base Qwen3-8B performance (proving it ignores order). If it DOES hurt, the pain point is weaker and we pivot.

**Estimated GPU hours**: 5h per domain × 4 domains × 5 variants = 100h (4090)

### Block 2: Need-Gate Validation
**Goal**: Prove the learned gate outperforms fixed alternatives.

| Experiment | Gate Setting | Domains | Seeds | Evidence Level |
|-----------|-------------|---------|-------|----------------|
| gate-learned | Learned (our method) | 4 | 5 | controlled |
| gate-fixed-05 | Fixed α=0.5 | 4 | 5 | controlled |
| gate-temporal-only | Fixed α=1.0 | 4 | 5 | controlled |
| gate-semantic-only | Fixed α=0.0 | 4 | 5 | controlled |
| gate-oracle | Oracle (α=1 when temporal correct) | 4 | 1 | diagnostic |

**Success criterion**: Learned gate > fixed-0.5 by ≥2% MRR on majority of domains.

**Estimated GPU hours**: 20h (gate training is CPU, only LLM eval needs GPU)

### Block 3: Component Ablation
**Goal**: Prove each component is necessary.

| Experiment | Ablated Component | Domains | Seeds | Evidence Level |
|-----------|------------------|---------|-------|----------------|
| abl-no-tdig | Random graph (no temporal structure) | 4 | 5 | controlled |
| abl-no-gate | No gate (fixed α=0.5) | 4 | 5 | controlled |
| abl-no-evidence-text | No evidence in LLM prompt | 4 | 5 | controlled |
| abl-no-lora | Base Qwen3-8B (no training) | 4 | 5 | controlled |
| abl-no-trap | No semantic trap penalty | 4 | 5 | controlled |
| abl-undirected | Undirected graph | 4 | 5 | controlled |
| abl-no-decay | Uniform edge weights | 4 | 5 | controlled |

**Success criterion**: Each ablation causes ≥1% MRR drop on majority of domains.

**Estimated GPU hours**: 7 variants × 4 domains × 5 seeds × 1h = 140h

### Block 4: Baseline Comparison (Main Table)
**Goal**: Compare against all 8 Pony official baselines.

| Experiment | Method | Domains | Seeds | Evidence Level |
|-----------|--------|---------|-------|----------------|
| main-ours | TGL-Rec (full) | 4 | 20 | official |
| main-ours-stage1 | TGL-Rec Stage 1 only | 4 | 20 | official |
| ctrl-history-lora | History-only LoRA | 4 | 5 | controlled |
| ctrl-evidence-lora | Evidence LoRA (no gate) | 4 | 5 | controlled |

Pony baselines (already complete, 1 seed each — reused):
- llm2rec, llmesr, llmemb, rlmrec, irllrec, elmrec, proex, promax

**Success criterion**: TGL-Rec beats best baseline on ≥3 domains with p<0.05.

**Estimated GPU hours**: 2 methods × 4 domains × 20 seeds × 2h = 320h

### Block 5: Mechanism Analysis
**Goal**: Understand WHEN and WHY the method works.

| Analysis | Description | Output |
|----------|-------------|--------|
| gate-activation | Plot α distribution by user transition pressure | Figure |
| semantic-trap-cases | Top-50 cases where trap detection helped | Table |
| failure-cases | Top-50 cases where our method fails | Table |
| domain-sensitivity | Which domains benefit most from temporal evidence? | Analysis |
| evidence-quality | Correlation between evidence confidence and ranking quality | Figure |

**Estimated GPU hours**: 10h (mostly analysis on saved predictions)

### Block 6: Robustness
**Goal**: Ensure results are not artifacts of specific settings.

| Experiment | Variation | Domains | Seeds |
|-----------|-----------|---------|-------|
| rob-tau-1d | Time window τ=1 day | 4 | 3 |
| rob-tau-30d | Time window τ=30 days | 4 | 3 |
| rob-topk-10 | Stage 2 K=10 | 4 | 3 |
| rob-topk-50 | Stage 2 K=50 | 4 | 3 |
| rob-lora-rank8 | LoRA rank=8 | 4 | 3 |
| rob-lora-rank32 | LoRA rank=32 | 4 | 3 |

**Estimated GPU hours**: 6 × 4 × 3 × 1.5h = 108h

## Total Estimated GPU Hours

| Block | Hours |
|-------|-------|
| Observation | 100 |
| Gate validation | 20 |
| Ablation | 140 |
| Main comparison | 320 |
| Mechanism | 10 |
| Robustness | 108 |
| **Total** | **~700h** |

On a single 4090: ~29 days continuous. With efficient scheduling and early stopping: ~2-3 weeks realistic.

## Execution Order

```
Week 1: Block 1 (Observation) — validate pain point exists
         If pain point confirmed → proceed
         If NOT confirmed → STOP and reformulate
Week 2: Block 2 (Gate) + Block 3 (Ablation) — validate components
         If gate helps → proceed to full training
         If NOT → simplify method, remove gate
Week 3-4: Block 4 (Main comparison) — 20-seed official runs
Week 4: Block 5 (Mechanism) + Block 6 (Robustness)
```

## Stage Gates

| Gate | Condition | Action if FAIL |
|------|-----------|----------------|
| G1 | Observation shows base Qwen3-8B ignores temporal order | Reformulate: maybe LLMs DO use temporal signals |
| G2 | Learned gate > fixed gate by ≥2% MRR | Simplify: use fixed gate, focus on evidence quality |
| G3 | Full method > best baseline on ≥2 domains | Analyze: which domains fail? Is the method domain-specific? |
| G4 | All ablations show significant degradation | Remove non-contributing components before paper |

## Falsification Criteria

The project should be ABANDONED or PIVOTED if:
1. Base Qwen3-8B is already sensitive to temporal order (shuffling hurts significantly)
2. No configuration of our method beats the best Pony baseline on any domain
3. The gate never activates differently for transition vs stable users

## Output Artifacts

- `outputs/experiments/observation/` — Block 1 results
- `outputs/experiments/gate_validation/` — Block 2 results
- `outputs/experiments/ablation/` — Block 3 results
- `outputs/experiments/main_comparison/` — Block 4 results (paper tables)
- `outputs/experiments/mechanism/` — Block 5 analysis
- `outputs/experiments/robustness/` — Block 6 results
- `outputs/tables/` — Exported paper-ready tables
- `outputs/figures/` — Generated figures

## Reproducibility Requirements

Every run must save:
- Resolved YAML config
- Git commit hash
- Random seed
- Environment (Python version, package versions, GPU)
- Full predictions.jsonl with event_id, user_id, item_id, score
- Metrics.json with all metrics
- Runtime.json with wall-clock time
