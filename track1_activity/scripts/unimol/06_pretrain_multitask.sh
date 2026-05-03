#!/usr/bin/env bash
# Multi-task pretrain: Uni-Mol v2 with 4 heads
#   log2fc_8p25 + log2fc_33 + pec50 + counter_pec50
# Stronger task signal vs PR #114 (log2fc-only). Same isolated pixi env.
#
# Inputs:
#   data/unimol/pretrain_multitask.csv (12,589 rows, NaN-safe per head)
# Output:
#   models/unimol_v2_multitask/exp/<ts>/model_<seed>.pth (checkpoint)
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-32}"
SEED="${SEED:-42}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_multitask.csv" ]]; then
    echo "ERROR: $PXR_REPO/data/unimol/pretrain_multitask.csv not found." >&2
    echo "Run 05_prepare_multitask_data.py first." >&2
    exit 1
fi

SAVE_DIR="${SAVE_DIR:-$PXR_REPO/models/unimol_v2_multitask}"
mkdir -p "$SAVE_DIR"

cat > /tmp/unimol_pretrain_multitask_invoke.py <<PYEOF
import argparse
from unimol_tools import MolTrain

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--save_dir', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
ap.add_argument('--epochs', type=int, default=$EPOCHS)
ap.add_argument('--batch', type=int, default=$BATCH)
ap.add_argument('--seed', type=int, default=$SEED)
args = ap.parse_args()

clf = MolTrain(
    task='regression',
    data_type='molecule',
    model_name='unimolv2',
    model_size=args.model_size,
    epochs=args.epochs,
    batch_size=args.batch,
    metrics='mae',
    save_path=args.save_dir,
    target_cols=['log2fc_8p25', 'log2fc_33', 'pec50', 'counter_pec50'],
    kfold=2,
    seed=args.seed,
)
clf.fit(data=args.data)
print('MolTrain (multitask) complete, checkpoint at:', args.save_dir)
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_pretrain_multitask_invoke.py \
    --data "$PXR_REPO/data/unimol/pretrain_multitask.csv" \
    --save_dir "$SAVE_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --seed "$SEED"
