# TGL-Rec Technical Design: Temporal Need-State Evidence for LLM Reranking

## 1. Core Claim (One Sentence)

LLM-based recommenders systematically underuse temporal transition signals because their attention mechanisms lack inherent sequence ordering; TGL-Rec fixes this by providing explicit temporal graph evidence through a learned need-gate that activates temporal signals only when the user's interaction state indicates a genuine transition need.

## 2. Differentiation from Closest Work

| Dimension | TGL-Rec (Ours) | CETRec [2507.03047] | G-Refer [2502.12586] | Temporal Awareness [2405.02778] |
|-----------|---------------|---------------------|----------------------|-------------------------------|
| Graph type | Temporal item-item TDIG | None | Static user-item CF | None |
| Evidence form | Structured temporal transitions | Counterfactual sequences | CF path explanations | Prompt templates |
| Gating mechanism | Learned need-gate from temporal state | None (uniform tuning) | None | None |
| Training signal | Transition prediction + listwise ranking | Counterfactual discrimination | Retrieval + generation | Zero-shot only |
| Semantic trap detection | Yes (penalizes high-sim low-transition) | No | No | No |
| Explainability | Factor-level score decomposition | No | Path-level explanation | No |

**Fundamental difference**: All competitors either (a) modify the LLM's internal temporal sensitivity or (b) retrieve static graph paths. We do neither — we construct a temporal item-item transition graph, compute a user-specific need-state, and use a learned gate to decide WHEN temporal evidence should override semantic similarity. The evidence is candidate-grounded and auditable.

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TGL-Rec Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Offline: Train-Only]                                           │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ TDIG Builder │───▶│ Edge Statistics  │                       │
│  │ (§4)         │    │ (transition prob,│                       │
│  └──────────────┘    │  PMI, lift, dir) │                       │
│                      └──────────────────┘                       │
│                                                                   │
│  [Per-User Inference]                                            │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Need-State   │───▶│ Evidence         │───▶│ Need-Gate    │  │
│  │ Encoder (§5) │    │ Retriever (§6)   │    │ (Learned, §7)│  │
│  └──────────────┘    └──────────────────┘    └──────────────┘  │
│                                                        │         │
│                                                        ▼         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 1: Evidence Scoring (all 101 candidates)           │   │
│  │ score_c = α_c · temporal_c + (1-α_c) · semantic_c       │   │
│  │         + recency_c - trap_penalty_c                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼ top-K                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 2: LLM Reranking (LoRA Qwen3-8B)                  │   │
│  │ Input: history + translated evidence + top-K candidates  │   │
│  │ Output: final ranking                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 4. TDIG: Temporal Directed Item Graph

### Definition

Given train-only interactions $\mathcal{D}_{\text{train}} = \{(u, i, t)\}$:

**Nodes**: All items $i \in \mathcal{I}$ appearing in train split.

**Directed edges**: For each user $u$ with consecutive interactions $(i, t_1)$ and $(j, t_2)$ where $0 < t_2 - t_1 < \tau$:
- Add directed edge $i \to j$

**Edge weight** (time-decayed transition frequency):
$$w_{ij} = \sum_{\substack{(u,i,t_1),(u,j,t_2) \in \mathcal{D}_{\text{train}} \\ 0 < t_2-t_1 < \tau}} \exp\left(-\lambda \cdot (t_2 - t_1)\right)$$

**Derived statistics** (per edge):
- Transition probability: $P(j|i) = w_{ij} / \sum_k w_{ik}$
- PMI: $\text{PMI}(i,j) = \log \frac{P(j|i)}{P(j)}$
- Lift: $\text{Lift}(i,j) = P(j|i) / P(j)$
- Direction asymmetry: $\text{DA}(i,j) = |P(j|i) - P(i|j)|$

**Time-window variant**: Edges weighted by co-occurrence within sliding windows of size $\tau_w$, capturing "items consumed together in a session."

### Hyperparameters
- $\tau$: maximum gap for transition edge (default: 7 days)
- $\lambda$: time decay rate (default: 1/86400, i.e., per-day decay)
- $\tau_w$: session window size (default: 1 hour)

### Leakage Prevention
- Graph built from train split ONLY
- No edges involving test/valid target items
- Provenance recorded: `constructed_from=train_only`

## 5. Need-State Encoder

### Motivation

Not all users benefit equally from temporal evidence. A user in "stable preference" mode (repeatedly buying the same category) needs less temporal guidance than a user in "transition" mode (exploring new categories after a life event). The need-state encoder captures this distinction.

### Formulation

For user $u$ with history $H_u = [(i_1, t_1), \ldots, (i_n, t_n)]$ sorted by time:

**Stable preference signal**:
$$\mathbf{s}_u = \frac{1}{n} \sum_{k=1}^{n} \mathbf{e}_{i_k}$$

where $\mathbf{e}_i$ is the item's category/text embedding (frozen, from item metadata).

**Recent drift vector**:
$$\mathbf{d}_u = \frac{1}{w} \sum_{k=n-w+1}^{n} \mathbf{e}_{i_k} - \mathbf{s}_u$$

**Transition pressure** (from TDIG out-degree of last item):
$$p_u = \frac{\sum_{j \in \text{out}(i_n)} w_{i_n j} \cdot \mathbb{1}[\text{cat}(j) \neq \text{cat}(i_n)]}{\sum_{j \in \text{out}(i_n)} w_{i_n j} + \epsilon}$$

**Temporal gap** (normalized time since last interaction):
$$g_u = \min\left(1, \frac{t_{\text{now}} - t_n}{\tau_{\text{max}}}\right)$$

**Category entropy** (diversity of recent interactions):
$$\text{ent}_u = -\sum_{c} p_c \log p_c, \quad p_c = \frac{|\{i_k : \text{cat}(i_k)=c, k > n-w\}|}{w}$$

**Need-state vector**:
$$\mathbf{n}_u = [\|\mathbf{d}_u\|, p_u, g_u, \text{ent}_u, n] \in \mathbb{R}^5$$

### Implementation Note
This is computed from train-only data and item metadata. No learned parameters in the encoder itself — learning happens in the need-gate (§7).

## 6. Evidence Retrieval

### Per-Candidate Evidence

For each candidate $c$ in the candidate set, retrieve from TDIG:

1. **Transition evidence**: All edges $i_k \to c$ where $i_k \in H_u$ (user's history items that transition to $c$)
2. **Time-window evidence**: Co-occurrence of $c$ with recent history items within session windows
3. **Contrastive evidence**: Items $c'$ that have high semantic similarity to $c$ but LOW transition support (semantic traps)
4. **Recency evidence**: How recently items similar to $c$ appeared in history

### Evidence Feature Vector

For candidate $c$, aggregate retrieved evidence into:
$$\mathbf{e}_c = [\text{trans}_c, \text{prob}_c, \text{pmi}_c, \text{lift}_c, \text{da}_c, \text{win}_c, \text{sem}_c, \text{trap}_c, \text{rec}_c, \text{conf}_c] \in \mathbb{R}^{10}$$

Where:
- $\text{trans}_c = \log(1 + \sum_{i_k \to c} w_{i_k c})$ — total transition support
- $\text{prob}_c = \max_{i_k \in H_u} P(c|i_k)$ — max transition probability from history
- $\text{pmi}_c = \max_{i_k \in H_u} \text{PMI}(i_k, c)$ — max PMI
- $\text{lift}_c = \max_{i_k \in H_u} \text{Lift}(i_k, c)$ — max lift
- $\text{da}_c = \text{mean}_{i_k \in H_u} \text{DA}(i_k, c)$ — mean direction asymmetry
- $\text{win}_c$ — time-window co-occurrence score
- $\text{sem}_c$ — semantic similarity to recent history (BM25 or embedding)
- $\text{trap}_c = \max(0, \text{sem}_c - \text{trans}_c)$ — semantic trap indicator
- $\text{rec}_c$ — recency of supporting evidence
- $\text{conf}_c$ — evidence confidence (diversity × volume)

## 7. Need-Gate (Learned)

### The Core Innovation

The need-gate is a lightweight learned function that decides, for each (user, candidate) pair, how much to trust temporal evidence vs semantic similarity:

$$\alpha_c = \sigma\left(\mathbf{w}^T [\mathbf{n}_u; \mathbf{e}_c; \mathbf{n}_u \odot \mathbf{e}_c] + b\right)$$

Where:
- $\mathbf{n}_u \in \mathbb{R}^5$ is the user's need-state
- $\mathbf{e}_c \in \mathbb{R}^{10}$ is the candidate's evidence features
- $\mathbf{n}_u \odot \mathbf{e}_c$ is element-wise interaction (after projection to same dim)
- $\mathbf{w} \in \mathbb{R}^{25}$, $b \in \mathbb{R}$ are learned parameters

### Why This Is Not Just Attention

Standard attention computes relevance between query and key. Our gate computes **when temporal evidence is trustworthy** — it's a meta-decision about evidence reliability, not content relevance. The gate is high when:
- User has high transition pressure (exploring new categories)
- Candidate has strong, diverse temporal support
- There's a semantic trap (high similarity but low transition)

The gate is low when:
- User is in stable preference mode
- Temporal evidence is sparse or unreliable
- Semantic similarity genuinely indicates preference

### Training Objective

**Positive examples**: (user, candidate) pairs where:
- candidate IS the ground-truth next item
- candidate has temporal evidence support (transition edges exist)

**Negative examples**: (user, candidate) pairs where:
- candidate is NOT the next item
- candidate has HIGH semantic similarity to history (semantic trap)

Loss: Binary cross-entropy weighted by evidence confidence:
$$\mathcal{L}_{\text{gate}} = -\sum_{(u,c)} \text{conf}_c \cdot [y \log \alpha_c + (1-y) \log(1-\alpha_c)]$$

### Parameter Count

Total: ~25 weights + 1 bias = 26 parameters. This is intentionally tiny — the gate should be a simple, interpretable decision boundary, not a black box.

## 8. Two-Stage Scoring

### Stage 1: Evidence-Based Scoring (All Candidates)

For all 101 candidates in the same-candidate protocol:

$$\text{score}_c = \alpha_c \cdot f_{\text{temp}}(\mathbf{e}_c) + (1 - \alpha_c) \cdot f_{\text{sem}}(\mathbf{e}_c) + f_{\text{rec}}(\mathbf{e}_c) - \beta \cdot \text{trap}_c$$

Where:
- $f_{\text{temp}}(\mathbf{e}_c) = \text{trans}_c + \text{prob}_c + \text{pmi}_c + \text{lift}_c + \text{da}_c + \text{win}_c$
- $f_{\text{sem}}(\mathbf{e}_c) = \text{sem}_c$
- $f_{\text{rec}}(\mathbf{e}_c) = \text{rec}_c$
- $\beta$ is the semantic trap penalty weight

### Stage 2: LLM Reranking (Top-K from Stage 1)

Select top-K candidates (K=20) from Stage 1. For each:

1. **Translate evidence to text** via GraphToTextTranslator:
   ```
   Candidate: "Harry Potter Book 5"
   Temporal evidence: 3 users who bought Book 4 within 7 days also bought Book 5
   (transition_prob=0.42, lift=8.3, direction: Book4→Book5 is 3x stronger than reverse)
   Need-state: User recently finished Book 4 (2 days ago), showing category exploration
   ```

2. **LoRA-tuned Qwen3-8B reranks** with prompt:
   - User history (last 10 items with timestamps)
   - Evidence summaries for top-K candidates
   - Instruction: rank candidates considering both preference and temporal need

### LoRA Training

- **Base model**: Qwen3-8B
- **Adapter**: LoRA rank=16, alpha=32, target modules: q_proj, v_proj
- **Training data**: Train split, same-candidate format
- **Loss**: ListMLE (listwise ranking loss)
- **Input**: History + evidence text + candidate list
- **Target**: Permutation with ground-truth item ranked first
- **Epochs**: 3, batch size 4, gradient accumulation 8
- **Validation**: Same-candidate valid split, early stopping on MRR

## 9. Ablation Matrix

Every component is independently switchable via config:

| Switch | ON (default) | OFF (ablation) | Tests |
|--------|-------------|----------------|-------|
| `use_tdig` | Full temporal graph | No graph (random evidence) | Is temporal structure needed? |
| `use_need_gate` | Learned gate | Fixed α=0.5 | Does adaptive gating help? |
| `gate_mode` | learned | temporal_only (α=1) / semantic_only (α=0) | Which signal dominates? |
| `use_evidence_text` | Translated evidence in prompt | No evidence in LLM prompt | Does explicit evidence help LLM? |
| `use_time_decay` | Exponential decay weights | Uniform weights | Does recency matter in graph? |
| `use_direction` | Directed edges | Undirected (symmetric) | Does transition direction matter? |
| `use_semantic_trap` | Trap penalty active | No penalty | Does trap detection help? |
| `use_lora` | LoRA fine-tuned | Base Qwen3-8B | Does training help? |
| `use_stage1` | Two-stage (evidence + LLM) | LLM only (all candidates) | Does pre-filtering help? |
| `evidence_mode` | Full (all types) | transition_only / window_only / semantic_only | Which evidence type matters? |

### Required Ablation Experiments (Minimum for Paper)

1. Full method vs no-gate (fixed α)
2. Full method vs no-TDIG (random graph)
3. Full method vs no-evidence-text (LLM without evidence)
4. Full method vs base Qwen3-8B (no LoRA)
5. Temporal-only vs semantic-only gate
6. Directed vs undirected graph
7. With vs without semantic trap penalty
8. Evidence type ablation (transition / window / semantic / all)

## 10. Evaluation Protocol

### Metrics
- MRR, HR@5, HR@10, HR@20, NDCG@5, NDCG@10, NDCG@20
- Coverage@K, longtail_coverage@K
- Parse success rate, candidate adherence rate

### Statistical Rigor
- 20+ seeds for paper results
- Paired t-test between our method and each baseline
- Bootstrap 95% CI for all reported metrics
- Effect size (Cohen's d) for significant improvements

### Domains
- Beauty (973 users, supplementary smaller-N)
- Books (10,000 users)
- Electronics (10,000 users)
- Movies (10,000 users)

### Baselines (8 Pony Official + Our Controls)
1. llm2rec (SASRec + LLM)
2. llmesr (SASRec + LLM enhanced)
3. llmemb (LLM embeddings)
4. rlmrec (RL + GraphCL)
5. irllrec (Intent-aware)
6. elmrec (Graph-enhanced)
7. proex (Profile-based)
8. promax (Profile-based, extended)
9. Base Qwen3-8B (no adapter, our control)
10. History-only LoRA (our control)
11. Temporal-evidence LoRA without gate (our control)

## 11. Implementation Plan

### Phase A: Need-Gate Training (Local + Server)
1. Extract need-state features from train data (local)
2. Extract evidence features for all (user, candidate) pairs (local)
3. Train logistic regression gate on train split (local, fast)
4. Validate gate quality: does high-α correlate with temporal-need users? (local)

### Phase B: Evidence-Augmented LoRA Training (Server)
1. Generate evidence text for all train examples (server, GPU for embedding)
2. Format as LoRA training data (history + evidence + candidates → ranking)
3. Train LoRA adapter with ListMLE loss (server, 4090 GPU)
4. Validate on valid split, tune K and evidence format

### Phase C: Full Pipeline Evaluation (Server)
1. Run observation: base Qwen3-8B on 4 domains (confirm pain point)
2. Run our method: Stage 1 scoring → Stage 2 LLM reranking
3. Run all ablations
4. Compare against 8 Pony baselines + 3 controls
5. Statistical tests, table export

### Phase D: Analysis & Paper Prep (Local)
1. Paired comparison with event-level diagnostics
2. Case studies: where does the gate activate? Where does it fail?
3. Semantic trap analysis: examples of high-sim low-transition candidates
4. Failure case documentation

## 12. Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|------------|
| Gate too simple (26 params) to capture complex patterns | Medium | If logistic fails, upgrade to 2-layer MLP (still <1000 params) |
| LoRA training doesn't converge | Low | Fallback: use Stage 1 only (no LLM reranking) |
| Evidence text confuses LLM | Medium | Ablate evidence format: JSON vs NL vs structured |
| Can't beat llm2rec on books (MRR 0.198) | Medium | Focus on domains where temporal signal is stronger |
| Reviewer says "just G-Refer + time" | Low | Ablation proves gate and TDIG are independently necessary |

## 13. What Makes This NOT Stitching

The method is NOT: G-Refer + timestamps + gating. Here's why:

1. **G-Refer** retrieves collaborative filtering paths (user→item→user→item). We retrieve temporal item-item transitions. Different graph, different semantics.

2. **CETRec** modifies the LLM's internal temporal sensitivity. We provide external evidence. Different mechanism entirely.

3. **The need-gate** is not a generic attention mechanism. It's a meta-decision about evidence reliability conditioned on user temporal state. No prior work computes "transition pressure" or "semantic trap" as gating signals.

4. **The two-stage design** is not just "retrieve then rerank." Stage 1 uses the learned gate to score ALL candidates efficiently; Stage 2 uses evidence TEXT to help the LLM reason. The gate and the LLM serve different roles.

5. **Ablation proof**: If removing any single component (TDIG, gate, evidence text, LoRA) significantly hurts performance, the contribution is the INTEGRATION with each part being necessary — not a bag of tricks where any subset works equally well.
