#!/usr/bin/env bash
# Phase A: log2fc FT on filtered (clean conformer) compounds.
# Codex 2026-05-02 priority: tests "noisy ETKDG conformers hurt FT" hypothesis.
#
# Differences from PR #114 02_pretrain_molv2.sh:
#   - Input: pretrain_log2fc_filtered.csv (84 sterically complex dropped)
#   - kfold=1 (test if multi-target broadcast bug from PR #114 is fixed;
#     fall back to kfold=2 if it errors)
#   - epochs=15 (PR #114 reported Pearson 0.09, likely converged early)
#   - batch=16 (env-required to avoid backward OOM at 16GB)
#
# Inputs:
#   data/unimol/pretrain_log2fc_filtered.csv (10751 rows, 2 log2fc heads)
# Output:
#   models/unimol_v2_log2fc_filtered/exp/<ts>/model_<seed>.pth
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-16}"
SEED="${SEED:-42}"
KFOLD="${KFOLD:-1}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_log2fc_filtered.csv" ]]; then
    echo "ERROR: pretrain_log2fc_filtered.csv not found." >&2
    echo "Run 11_prepare_filtered_log2fc.py first." >&2
    exit 1
fi

SAVE_DIR="${SAVE_DIR:-$PXR_REPO/models/unimol_v2_log2fc_filtered}"
mkdir -p "$SAVE_DIR"

cat > /tmp/unimol_pretrain_log2fc_filtered.py <<PYEOF
import argparse
from unimol_tools import MolTrain

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--save_dir', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
ap.add_argument('--epochs', type=int, default=$EPOCHS)
ap.add_argument('--batch', type=int, default=$BATCH)
ap.add_argument('--seed', type=int, default=$SEED)
ap.add_argument('--kfold', type=int, default=$KFOLD)
args = ap.parse_args()

clf = MolTrain(
    task='regression',
    data_type='molecule',
    model_name='unimolv2',
    model_size=args.model_size,
    epochs=args.epochs,
    batch_size=args.batch,
    metrics='none',  # skip sklearn val-metric (NaN-unsafe in newer sklearn)
    save_path=args.save_dir,
    target_cols=['log2fc_8p25', 'log2fc_33'],
    kfold=args.kfold,
    seed=args.seed,
)
clf.fit(data=args.data)
print('MolTrain (filtered log2fc) complete, checkpoint at:', args.save_dir)
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_pretrain_log2fc_filtered.py \
    --data "$PXR_REPO/data/unimol/pretrain_log2fc_filtered.csv" \
    --save_dir "$SAVE_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --seed "$SEED" \
    --kfold "$KFOLD"
