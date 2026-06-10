#!/usr/bin/env bash
# CC-PACE beauty experiment runner (server). Launches ONLY when the GPU has
# enough free memory, so it never preempts another job (e.g. Pony).
#
# Usage: bash run_cc_pace_beauty.sh [limit]      (limit=0 -> all 973 users)
# Runs the full method + the by-design ablations and writes one JSON each.
# Go/kill thresholds: docs/method_v2_decision_CC-PACE.md.
set -u
PROJ=/home/ajifang/projects/TGL-Rec
PY=/home/ajifang/miniconda3/envs/tglrec/bin/python
TASK="$PROJ/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_test_same_candidate/ranking_test.jsonl"
OUTDIR="$PROJ/outputs/cc_pace_beauty"
SCRIPT="$PROJ/scripts/cc_pace_beauty.py"
LIMIT="${1:-0}"

need_mb=17000
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${free_mb} MiB (need >= ${need_mb})"
if [ "$free_mb" -lt "$need_mb" ]; then
  echo "INSUFFICIENT GPU — not launching (another job is using the card)."
  exit 3
fi

cd "$PROJ" || exit 1
mkdir -p "$OUTDIR"
for variant in full text_only no_residualizer no_shrinkage rich_residualizer; do
  echo "=== CC-PACE beauty variant: $variant ==="
  "$PY" "$SCRIPT" --task "$TASK" --out "$OUTDIR/${variant}.json" --limit "$LIMIT" --variant "$variant"
done
echo "done -> $OUTDIR (compare full vs ablations; check beats_sota_ndcg10 + panel/CF ablation gaps)"
