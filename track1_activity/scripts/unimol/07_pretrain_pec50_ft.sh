#!/usr/bin/env bash
# Direct pEC50 fine-tune: Uni-Mol v2 with pec50 only target.
# Smaller data (4140 rows from train_activity), fewer epochs to avoid overfit.
# Codex 2026-05-02 Phase 2: log2fc-bypass, task-aligned.
#
# Inputs:
#   data/unimol/pretrain_pec50.csv (4140 rows, SMILES + pec50)
# Output:
#   models/unimol_v2_pec50_ft/exp/<ts>/model_<seed>.pth
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
EPOCHS="${EPOCHS:-15}"   # smaller dataset, lower epochs to avoid overfit
BATCH="${BATCH:-32}"
SEED="${SEED:-42}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_pec50.csv" ]]; then
    echo "ERROR: $PXR_REPO/data/unimol/pretrain_pec50.csv not found." >&2
    echo "Run 08_prepare_pec50_data.py first." >&2
    exit 1
fi

SAVE_DIR="${SAVE_DIR:-$PXR_REPO/models/unimol_v2_pec50_ft}"
mkdir -p "$SAVE_DIR"

cat > /tmp/unimol_pretrain_pec50_invoke.py <<PYEOF
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
    target_cols=['pec50'],
    kfold=2,
    seed=args.seed,
)
clf.fit(data=args.data)
print('MolTrain (pec50 ft) complete, checkpoint at:', args.save_dir)
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_pretrain_pec50_invoke.py \
    --data "$PXR_REPO/data/unimol/pretrain_pec50.csv" \
    --save_dir "$SAVE_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --seed "$SEED"
