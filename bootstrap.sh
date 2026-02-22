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

CMD="cd $WORKDIR && source .venv/bin/activate && \
PYTHONPATH=src python -m neurofin.train \
  --data-root $DATA_ROOT \
  --output-dir $OUT_DIR \
  --use-4bit \
  --subjects UTS01 \
  --context-tokens 256 \
  --n-test-stories 4 \
  --feature-cache-dir $CACHE_DIR \
  --batch-size 32 2>&1 | tee $LOG_DIR/train_uts01.log"

tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f $LOG_DIR/train_uts01.log"
echo "GPU check: watch -n 2 nvidia-smi"
