#!/usr/bin/env bash
# CC-PACE beauty experiment runner (server). Launches ONLY when the GPU has
# enough free memory, so it never preempts another job (e.g. Pony).
#
# Usage: bash run_cc_pace_beauty.sh [limit]      (limit=0 -> all 973 users)
# Steps: (1) bootstrap the frozen task file from Pony's external_tasks (read-only
# copy, one-time), (2) build the frozen CF artifacts + profile slots on CPU if
# missing, (3) run the full method + the by-design ablations, one JSON each.
# Go/kill thresholds: docs/method_v2_decision_CC-PACE.md.
set -u
PROJ=/home/ajifang/projects/TGL-Rec
PY=/home/ajifang/miniconda3/envs/tglrec-lora/bin/python
PONY_TASK=/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_test_same_candidate/ranking_test.jsonl
TASK="$PROJ/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_test_same_candidate/ranking_test.jsonl"
TRAIN="$PROJ/data/domains/beauty/train_interactions.jsonl"
OUTDIR="$PROJ/outputs/cc_pace_beauty"
SCRIPT="$PROJ/scripts/cc_pace_beauty.py"
CF_ART="$OUTDIR/cf_artifacts.json"
PROFILES="$OUTDIR/profiles.json"
LIMIT="${1:-0}"

# --- one-time bootstrap: copy the frozen protocol task file (Pony stays read-only) ---
if [ ! -f "$TASK" ]; then
  echo "bootstrapping task file from Pony external_tasks (read-only source)"
  mkdir -p "$(dirname "$TASK")"
  cp "$PONY_TASK" "$TASK" || { echo "FATAL: cannot copy task file"; exit 1; }
fi

mkdir -p "$OUTDIR"

# --- CPU data prep (no GPU needed; idempotent) ---
if [ ! -f "$CF_ART" ]; then
  echo "=== building frozen CF artifacts (SASRec, CPU) ==="
  "$PY" "$PROJ/scripts/build_cc_pace_cf_artifacts.py" \
    --train-interactions "$TRAIN" --task "$TASK" --out "$CF_ART" || exit 1
fi
if [ ! -f "$PROFILES" ]; then
  echo "=== building profile slots (train-history titles, CPU) ==="
  "$PY" "$PROJ/scripts/build_cc_pace_profiles.py" \
    --task "$TASK" --out "$PROFILES" || exit 1
fi

need_mb=17000
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${free_mb} MiB (need >= ${need_mb})"
if [ "$free_mb" -lt "$need_mb" ]; then
  echo "INSUFFICIENT GPU — not launching (another job is using the card)."
  exit 3
fi

cd "$PROJ" || exit 1
for variant in full text_only no_residualizer no_shrinkage rich_residualizer; do
  echo "=== CC-PACE beauty variant: $variant ==="
  "$PY" "$SCRIPT" --task "$TASK" --out "$OUTDIR/${variant}.json" --limit "$LIMIT" \
    --variant "$variant" --cf-artifacts "$CF_ART" --profiles "$PROFILES"
done
echo "done -> $OUTDIR (compare full vs ablations; check beats_sota_ndcg10 + panel/CF ablation gaps)"
