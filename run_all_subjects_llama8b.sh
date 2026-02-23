#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$HOME/neurofin"
DATA_ROOT="$WORKDIR/ds003020"
CACHE_DIR="$WORKDIR/feature_cache_llama8b_L24-31_ctx512"
LOG_DIR="$WORKDIR/logs"
PYTHON="$WORKDIR/.venv/bin/python3"
LAYER_INDICES="24,25,26,27,28,29,30,31"
CTX=512
MODEL="meta-llama/Llama-3.1-8B"

mkdir -p "$LOG_DIR" "$CACHE_DIR"
cd "$WORKDIR"

run_subject() {
  local SUBJ=$1
  local OUT_DIR="$WORKDIR/artifacts_llama8b_L24-31_ctx512_${SUBJ}"
  local LOG="$LOG_DIR/llama8b_L24-31_ctx512_${SUBJ}.log"
  echo "=== Starting ${SUBJ} at $(date) ===" | tee -a "$LOG"
  PYTHONPATH=src "$PYTHON" -m neurofin.train \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT_DIR" \
    --model-name "$MODEL" \
    --subjects "$SUBJ" \
    --layer-indices "$LAYER_INDICES" \
    --context-tokens "$CTX" \
    --n-test-stories 4 \
    --feature-cache-dir "$CACHE_DIR" \
    --batch-size 16 2>&1 | tee -a "$LOG"
  echo "=== Finished ${SUBJ} at $(date) ===" | tee -a "$LOG"
}

check_gate() {
  local SUBJ=$1
  local OUT_DIR="$WORKDIR/artifacts_llama8b_L24-31_ctx512_${SUBJ}"
  "$PYTHON" - <<PY
import json, sys
m = json.load(open("$OUT_DIR/training_metrics.json"))
r_top5 = m.get("mean_top5pct_corr", 0)
print(f"${SUBJ} mean_top5pct_corr: {r_top5:.4f}")
if r_top5 < 0.08:
    sys.exit(f"${SUBJ} top5pct={r_top5:.4f} below 0.08 — check before continuing.")
PY
}

# UTS01: full run (feature extraction + ridge, ~60 min)
# Skip if already done
if [ -f "$WORKDIR/artifacts_llama8b_L24-31_ctx512_UTS01/training_metrics.json" ]; then
  echo "UTS01 already complete, skipping extraction."
else
  run_subject UTS01
fi
check_gate UTS01

# UTS02-UTS08: cache hits (~10 min each, ridge only)
for SUBJ in UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08; do
  run_subject "$SUBJ"
done

echo ""
echo "=== All subjects complete. Results: ==="
for SUBJ in UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08; do
  OUT_DIR="$WORKDIR/artifacts_llama8b_L24-31_ctx512_${SUBJ}"
  if [ -f "$OUT_DIR/training_metrics.json" ]; then
    "$PYTHON" -c "
import json
m = json.load(open('$OUT_DIR/training_metrics.json'))
print('${SUBJ}: top5pct={:.4f}  mean={:.4f}'.format(m['mean_top5pct_corr'], m['mean_corr']))
"
  fi
done
