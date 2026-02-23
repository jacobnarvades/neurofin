#!/usr/bin/env bash
set -euo pipefail

# ---------- user vars ----------
REPO_URL="https://github.com/jacobnarvades/neurofin.git"
WORKDIR="$HOME/neurofin"
DATA_ROOT="$WORKDIR/ds003020"
OUT_DIR="$WORKDIR/artifacts_uts01"
CACHE_DIR="$WORKDIR/feature_cache"
LOG_DIR="$WORKDIR/logs"
SESSION_NAME="neurofin_train"
# ------------------------------

mkdir -p "$LOG_DIR"

sudo apt-get update
sudo apt-get install -y git rsync tmux awscli python3-venv python3-pip

if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install protobuf
pip install bitsandbytes

mkdir -p "$DATA_ROOT/derivative/preprocessed_data/UTS01"
mkdir -p "$DATA_ROOT/derivative/TextGrids"

aws s3 sync --no-sign-request \
  s3://openneuro.org/ds003020/derivative/preprocessed_data/UTS01 \
  "$DATA_ROOT/derivative/preprocessed_data/UTS01"

aws s3 sync --no-sign-request \
  s3://openneuro.org/ds003020/derivative/TextGrids \
  "$DATA_ROOT/derivative/TextGrids"

for SUBJ in UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08; do
  aws s3 sync --no-sign-request \
    s3://openneuro.org/ds003020/derivative/preprocessed_data/$SUBJ \
    "$DATA_ROOT/derivative/preprocessed_data/$SUBJ"
done

# Download pycortex-db transforms for all subjects (contains mask_thick.nii.gz
# needed for flat .hf5 voxel -> Schaefer parcel mapping during ROI aggregation).
for SUBJ in UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08; do
  aws s3 sync --no-sign-request \
    s3://openneuro.org/ds003020/derivative/pycortex-db/$SUBJ/transforms \
    "$DATA_ROOT/derivative/pycortex-db/$SUBJ/transforms"
done

python - <<'PY'
from pathlib import Path
hf5 = list(Path("ds003020/derivative/preprocessed_data/UTS01").glob("*.hf5"))
tg = list(Path("ds003020/derivative/TextGrids").glob("*.TextGrid"))
print("hf5_files:", len(hf5))
print("textgrids:", len(tg))
if len(hf5) == 0 or len(tg) == 0:
    raise SystemExit("Dataset sanity check failed: hf5 or TextGrid count is 0.")
PY

PYTHONPATH=src ./.venv/bin/python -m neurofin.train --data-root "$DATA_ROOT" --dry-split --subjects UTS01

# Write full training sequence to a script for tmux.
# Variables expanded here (paths hardcoded); \${SUBJ} left for the loop in the generated script.
cat > "$WORKDIR/run_all_training.sh" <<TRAINEOF
#!/usr/bin/env bash
set -euo pipefail
cd $WORKDIR
source .venv/bin/activate

# ---- Phase 1: UTS01 ----
PYTHONPATH=src python -m neurofin.train \\
  --data-root $DATA_ROOT \\
  --output-dir $OUT_DIR \\
  --subjects UTS01 \\
  --context-tokens 256 \\
  --n-test-stories 4 \\
  --feature-cache-dir $CACHE_DIR \\
  --batch-size 16 2>&1 | tee $LOG_DIR/train_uts01.log

# ---- Validation gate ----
# mean_top5pct_corr is the mean over the top 5% of voxels by held-out
# test-story correlation — comparable to the per-subject metrics reported
# in encoding-model literature (Antonello et al. 2023: ~0.15-0.25).
# mean_corr (all voxels) is diluted by non-language subcortical regions
# and is not the right gate metric.
python - <<'PY'
import json, sys
m = json.load(open("$OUT_DIR/training_metrics.json"))
r_all  = m.get("mean_corr", 0)
r_top5 = m.get("mean_top5pct_corr", 0)
print(f"UTS01 mean_corr (all voxels): {r_all:.4f}")
print(f"UTS01 mean_top5pct_corr:      {r_top5:.4f}")
if r_top5 < 0.08:
    sys.exit(f"mean_top5pct_corr={r_top5:.4f} below threshold 0.08 — stopping before multi-subject training.")
PY

# ---- Phase 2: UTS02-UTS08 (cache hits — ridge only) ----
for SUBJ in UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08; do
  PYTHONPATH=src python -m neurofin.train \\
    --data-root $DATA_ROOT \\
    --output-dir $WORKDIR/artifacts_\${SUBJ} \\
    --subjects \${SUBJ} \\
    --context-tokens 256 \\
    --n-test-stories 4 \\
    --feature-cache-dir $CACHE_DIR \\
    --batch-size 16 2>&1 | tee $LOG_DIR/train_\${SUBJ}.log
done
TRAINEOF

chmod +x "$WORKDIR/run_all_training.sh"
tmux new-session -d -s "$SESSION_NAME" "bash $WORKDIR/run_all_training.sh"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f $LOG_DIR/train_uts01.log"
echo "GPU check: watch -n 2 nvidia-smi"
