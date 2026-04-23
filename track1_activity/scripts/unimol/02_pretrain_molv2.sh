#!/usr/bin/env bash
# Pretrain Uni-Mol v2 on log2_fc labels using unimol_tools.MolTrain.
# Runs in the isolated Uni-Mol pixi env.
#
# Inputs:
#   data/unimol/pretrain_labeled_clean.csv (SMILES + log2fc_8p25 + log2fc_33)
# Output:
#   models/unimol_v2_log2fc/exp/<ts>/model_<seed>.pth (checkpoint)
#
# Usage:
#   bash track1_activity/scripts/unimol/02_pretrain_molv2.sh [--smoke]
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-32}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_labeled_clean.csv" ]]; then
    echo "ERROR: $PXR_REPO/data/unimol/pretrain_labeled_clean.csv not found. Run 01_prepare_log2fc_data.py first." >&2
    exit 1
fi

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE="--epochs 2"
    EPOCHS=2
fi

SAVE_DIR="$PXR_REPO/models/unimol_v2_log2fc"
mkdir -p "$SAVE_DIR"

cat > /tmp/unimol_pretrain_invoke.py <<PYEOF
import argparse
from unimol_tools import MolTrain

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--save_dir', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
ap.add_argument('--epochs', type=int, default=$EPOCHS)
ap.add_argument('--batch', type=int, default=$BATCH)
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
    target_cols=['log2fc_8p25', 'log2fc_33'],
    kfold=2,  # minimum valid CV avoiding kfold=1 multi-target broadcast bug
)
clf.fit(data=args.data)
print('MolTrain complete, checkpoint at:', args.save_dir)
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_pretrain_invoke.py \
    --data "$PXR_REPO/data/unimol/pretrain_labeled_clean.csv" \
    --save_dir "$SAVE_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH"
