#!/usr/bin/env bash
# Encode 13136 compounds via DrugCLIP molecule encoder (6-fold ensemble).
# Output: HDF5 with mol_reps shape (13136, 768).
#
# Runs inside DrugCLIP isolated pixi env. Must be invoked with cwd = repo root
# because encode_mols_multi_folds() looks for ./data/model_weights/6_folds/fold_*.pt
# relative paths.
set -euo pipefail

DRUGCLIP_REPO="${DRUGCLIP_REPO:-$HOME/ghq/github.com/THU-ATOM/Drug-The-Whole-Genome}"
PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
MOL_LMDB="${MOL_LMDB:-$DRUGCLIP_REPO/data/pxr_compounds.lmdb}"
SAVE_DIR="${SAVE_DIR:-$PXR_REPO/data/drugclip_embed}"
BATCH="${BATCH:-128}"
DEVICE="${DEVICE:-0}"

mkdir -p "$SAVE_DIR"

if [[ ! -f "$MOL_LMDB" ]]; then
    echo "ERROR: $MOL_LMDB not found. Run 01_smiles_to_lmdb.py first." >&2
    exit 1
fi

cd "$DRUGCLIP_REPO"

# encode_mols.py expects positional args: data_path (unused for our case),
# dict_path. We pass empty and "./dict".
CUDA_VISIBLE_DEVICES=$DEVICE \
pixi run --manifest-path "$DRUGCLIP_REPO/pixi.toml" python ./unimol/encode_mols.py \
    --user-dir ./unimol \
    "" "./dict" --valid-subset test \
    --results-path "$SAVE_DIR" \
    --num-workers 0 --ddp-backend=c10d --batch-size "$BATCH" \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --max-pocket-atoms 256 \
    --seed 1 \
    --log-interval 100 --log-format simple \
    --mol-path "$MOL_LMDB" \
    --save-dir "$SAVE_DIR" \
    --start 0 --end 14000 --write-h5
