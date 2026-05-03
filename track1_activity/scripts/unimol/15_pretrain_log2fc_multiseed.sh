#!/usr/bin/env bash
# Multi-seed pretrain: same setup as PR #114 (pretrain_labeled_clean.csv,
# 9444 rows × 30 epochs × kfold=2 × batch=16 × 84m), only differs by seed.
# Codex 2026-05-02 priority #1: variance reduction via 5-seed ensemble of
# Uni-Mol v2 log2fc FT.
#
# Usage:
#   SEED=43 SAVE_DIR=models/unimol_v2_log2fc_seed43 \
#     bash 15_pretrain_log2fc_multiseed.sh
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-16}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-$PXR_REPO/models/unimol_v2_log2fc_seed${SEED}}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_labeled_clean.csv" ]]; then
    echo "ERROR: pretrain_labeled_clean.csv not found." >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

cat > /tmp/unimol_pretrain_multiseed_invoke.py <<PYEOF
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
    target_cols=['log2fc_8p25', 'log2fc_33'],
    kfold=2,
    seed=args.seed,
)
clf.fit(data=args.data)
print(f'MolTrain seed={args.seed} complete, ckpt at: {args.save_dir}')
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_pretrain_multiseed_invoke.py \
    --data "$PXR_REPO/data/unimol/pretrain_labeled_clean.csv" \
    --save_dir "$SAVE_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --seed "$SEED"
