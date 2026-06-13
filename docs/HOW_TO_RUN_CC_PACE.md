# CC-PACE — How To Run (handoff guide for any agent)

This is the single entry point for running CC-PACE experiments and writing the paper.
Everything is implemented; an agent should be able to go straight to experiments. Read
`docs/method_v2_decision_CC-PACE.md` for the why, this file for the how.

## TL;DR pipeline
1. Baselines are DONE and frozen: `data/pony_official_baselines/` (8 domains × 8 official baselines,
   64-row master table). Beauty SOTA bar to beat = **promax NDCG@10 = 0.1506**.
2. Method = **CC-PACE** (`src/llm4rec/methods/cc_pace/`, ranker `llm4rec.rankers.CCPaceRanker`).
3. Run **beauty first** (zero-shot probe → ablations → LoRA if GO) → if SOTA, roll to the other 7
   domains. If not SOTA → re-run the 3-seat ARIS discussion, redesign, re-run beauty until SOTA
   (see `multi-agent-discussion-rule` in global memory).
4. After the performance table: do the 3 required experiments in `docs/paper_followup_experiments.md`
   (observation / ablation / hyperparameter) + an overview figure → submit.

## Discipline (non-negotiable; see global memory feedback_local_server_alignment)
- Experiments run on the SERVER; commit/push only from LOCAL.
- After each server run, scp lightweight evidence (metrics JSON/CSV, provenance) back to local;
  heavy artifacts (scores, checkpoints) stay server-side.
- Don't stop unless: server disk full / API failure / unfixable review reject / user says pause.
- Every chunk of work → update agentmemory + local docs + README.

## Code map (`src/llm4rec/methods/cc_pace/`)
| file | role |
|------|------|
| `config.py` | `CCPaceConfig` — every hyperparameter + `use_*` ablation switches |
| `schema.py` | unified candidate schema, label randomization (exchangeability), profile slots |
| `judge.py` | forced-choice listwise evidence extraction (`EvidenceModel` protocol) |
| `hf_judge.py` | concrete frozen Qwen3-8B model (server/GPU only; lazy torch import) |
| `cf_conditioning.py` | frozen CF as conditioning σ-field: evidence tokens + nuisance coords |
| `residualizer.py` | symmetric-LOO isotonic (additive) + rich (ceiling-test) residualizers |
| `conformal.py` | per-candidate conformal p, dual-null merge, James-Stein shrinkage |
| `statistic.py` | orchestrator: evidence → residual → shrink → ranked T_u (+ p_pop) |
| `llm4rec.rankers.cc_pace.CCPaceRanker` | BaseRanker entry point for the eval harness |
| `llm4rec.trainers.cc_pace_trainer` | Plackett-Luce loss + dCor penalty + training plan |

## Running the beauty experiment

### Local CPU plumbing/CI (mock judge — NOT for reporting)
```bash
python scripts/cc_pace_beauty.py --task <ranking_test.jsonl> --out outputs/cc_pace_beauty/full.json --mock --variant full
```

### Server zero-shot kill test (frozen Qwen3-8B, no training)
```bash
# only launches if >=17GB GPU free (never preempts another job)
bash scripts/run_cc_pace_beauty.sh 0        # 0 = all 973 users; small int = sanity subset
```
This runs `full` + the ablations `text_only`, `no_residualizer`, `no_shrinkage`, `rich_residualizer`
and writes one JSON each under `outputs/cc_pace_beauty/`.

### Resuming `full` after a GPU OOM
If `full.json.per_user.jsonl` exists but `full.json`/`go_verdict.json` is missing, do not rerun all
variants. Sync the latest branch, verify the HF judge equivalence test in the server conda env, then
resume only `full` from the per-user checkpoint and run the verdict script. The HF judge uses a
read-only expanded-cache wrapper after the 2026-06-13 KV-cache OOM; it avoids Transformers
`DynamicCache.update` materializing prompt-sized cache copies while preserving label logprob math.

```bash
cd /home/ajifang/projects/TGL-Rec
/home/ajifang/miniconda3/envs/tglrec-lora/bin/python -m pytest tests/unit/test_hf_judge_equivalence.py -q

OUTDIR=/home/ajifang/projects/TGL-Rec/outputs/cc_pace_beauty
TASK=/home/ajifang/projects/TGL-Rec/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_test_same_candidate/ranking_test.jsonl
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free_mb" -ge 43000 ] || { echo "GPU not clean: ${free_mb} MiB free"; exit 3; }
cp -a "$OUTDIR/full.json.per_user.jsonl" "$OUTDIR/full.json.per_user.jsonl.pre_resume_$(date +%Y%m%d_%H%M%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
/home/ajifang/miniconda3/envs/tglrec-lora/bin/python scripts/cc_pace_beauty.py \
  --task "$TASK" --out "$OUTDIR/full.json" --limit 0 --variant full \
  --cf-artifacts "$OUTDIR/cf_artifacts.json" --profiles "$OUTDIR/profiles.json" \
  > "$OUTDIR/full_resume_safe_$(date +%Y%m%d_%H%M%S).log" 2>&1
/home/ajifang/miniconda3/envs/tglrec-lora/bin/python scripts/cc_pace_go_verdict.py \
  --dir "$OUTDIR" --out "$OUTDIR/go_verdict.json"
```

For long resumes on a memory-fragmenting GPU, prefer chunked limits over one monolithic 973-user
process. The driver skips users already present in `full.json.per_user.jsonl`, so limits can advance
monotonically (for example `125, 150, ..., 950, 973`) and each Python process releases GPU memory
before the next chunk:

```bash
for limit in $(seq 125 25 950) 973; do
  python scripts/cc_pace_beauty.py \
    --task "$TASK" --out "$OUTDIR/full.json" --limit "$limit" --variant full \
    --cf-artifacts "$OUTDIR/cf_artifacts.json" --profiles "$OUTDIR/profiles.json"
done
python scripts/cc_pace_go_verdict.py --dir "$OUTDIR" --out "$OUTDIR/go_verdict.json"
```

### Go / kill (vs SOTA bar 0.1506) — from docs/method_v2_decision_CC-PACE.md
- **Zero-shot probe** (no training): if the residualized T carries no signal vs popularity → revisit
  before spending GPU on LoRA.
- **GO to LoRA:** zero-shot NDCG@10 ≥ 0.13 AND the CF-token ablation (`full` vs `text_only`) shows a
  positive gap.
- **STRONG GO / reportable:** post-LoRA NDCG@10 ≥ 0.1506 with paired-bootstrap p<0.05 over users, AND
  panel-corruption drops ≥30%, AND `text_only` falls below the CF baseline (proves the mechanism).
- **KILL / reframe:** only wins on popularity-heavy slice; or conformal coverage breaks; or T→0 under
  `rich_residualizer` and only late-fusion works → reposition as "panel-exchangeable calibration of a
  hybrid score" (honest fallback).

## Data-prep seams — WIRED (2026-06-12; nothing left before a real run)
Both seams are now implemented, tested (`tests/unit/test_cc_pace_artifacts.py`, 8 tests incl. a
torch end-to-end builder test), and integrated into the driver + server runner:

1. **CF provider artifacts** — `scripts/build_cc_pace_cf_artifacts.py` trains the repo's small
   SASRec on `data/domains/<domain>/train_interactions.jsonl` (TRAIN only; coverage check showed
   positives 95.1% / negatives 98.5% in-vocab, so no seen-ness leak) and writes one JSON with
   `user_scores` (panel-z-scored), `item_neighbors` (top-5 CF-embedding cosine neighbour titles),
   `item_clusters` (KMeans), plus `item_popularity` + `item_category` (title-lexicon facet) so the
   residualizer's `log_pop`/`facet_bucket` nuisances are real. Load with
   `cf_conditioning.load_cf_artifacts` + `provider_from_artifacts`.
2. **Profile slots** — `scripts/build_cc_pace_profiles.py` aggregates each user's TRAIN-history
   titles via `methods/cc_pace/text_facets.py` (beauty lexicons: category/routine/concern/
   ingredient + brand heuristic) into the schema's profile slots. `price_band` is omitted (no price
   data in the task files; the renderer skips empty slots).

The driver takes `--cf-artifacts` and `--profiles`; `scripts/run_cc_pace_beauty.sh` bootstraps the
frozen task file from Pony's external_tasks (read-only copy), builds both artifacts on CPU if
missing, then runs all variants with them attached. `text_only` still ablates CF via config switches
(zero code-path change — the non-stitch proof is intact).

## LoRA training (after a GO)
`llm4rec.trainers.cc_pace_trainer.build_training_plan(cfg)` emits the declarative plan; the loss is
`plackett_luce_top1_loss` (+ optional `distance_correlation` popularity penalty). Reuse the repo's
existing HF LoRA loop (`llm4rec.trainers.lora` / `lora_sft`) to apply the adapter, then evaluate with
the SAME driver via `--adapter <path>` (no `--mock`). Train panels come from TRAIN interactions only
(pseudo-held-out positive + popularity-matched & intent-matched negatives); split-conformal = train on
fold A, calibrate on disjoint fold B. The test panel is never seen in training.

## Rolling to the other 7 domains
Once beauty is SOTA, the same ranker + driver run unchanged on sports/toys/home/tools/books/
electronics/movies — point `--task` at each domain's `ranking_test.jsonl` and compare against that
domain's rows in `data/pony_official_baselines/baseline_comparison_8domains.csv`. Run domains serially
(server disk). Each domain's numbers + the ablations populate the paper's main table.

## Tests
`python -m pytest tests/unit/test_cc_pace_core.py tests/unit/test_cc_pace_ranker.py -q` (CPU, fast):
conformal validity, residualizer popularity-removal, shrinkage, schema exchangeability, end-to-end
ranker (mock judge), PL loss/grad.
