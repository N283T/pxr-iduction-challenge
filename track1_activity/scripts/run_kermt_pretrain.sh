#!/bin/bash
# KERMT continued-pretrain on single_concentration log2_fc.
#
# Runs in the KERMT-local pixi env (separate from main PXR pixi env).
# Inputs (produced by prepare_kermt_pretrain_csv.py):
#   data/kermt/pretrain_train.csv   (smiles, log2fc_8p25, log2fc_33)
#   data/kermt/pretrain_val.csv
# Output:
#   models/kermt/pretrain/fold_0/model_0/model.pt  (KERMT's default layout)
#
# Pass --epochs 2 as first arg to run a smoke test.
#
# NOTE: NVIDIA-Digital-Bio/KERMT's main.py lives at the repo root (not code/),
# so we invoke it from $KERMT_REPO directly.
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
KERMT_REPO="${KERMT_REPO:-$HOME/ghq/github.com/NVIDIA-Digital-Bio/KERMT}"
EPOCHS="${1:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# SEED env var: pretrain seed. Default 0 (KERMT default = original
# production checkpoint). Multi-seed ensembling: SEED=43, 44, ...
SEED="${SEED:-0}"
# SAVE_DIR env var: ckpt output. Default = canonical production path
# for SEED=0, seed-suffixed otherwise.
if [[ -z "${SAVE_DIR:-}" ]]; then
    if [[ "$SEED" -eq 0 ]]; then
        SAVE_DIR="$PXR_REPO/models/kermt/pretrain"
    else
        SAVE_DIR="$PXR_REPO/models/kermt/pretrain_seed${SEED}"
    fi
fi

if [[ ! -f "$KERMT_REPO/main.py" ]]; then
    echo "ERROR: KERMT repo not found at $KERMT_REPO (override with KERMT_REPO=...)" >&2
    exit 1
fi
if [[ ! -f "$PXR_REPO/data/kermt/pretrain_train.csv" ]]; then
    echo "ERROR: pretrain_train.csv not found. Run prepare_kermt_pretrain_csv.py first." >&2
    exit 1
fi
if [[ ! -f "$PXR_REPO/data/kermt/pretrain_val.csv" ]]; then
    echo "ERROR: pretrain_val.csv not found. Run prepare_kermt_pretrain_csv.py first." >&2
    exit 1
fi
if [[ ! -f "$PXR_REPO/models/kermt/grover_base.pt" ]]; then
    echo "ERROR: grover_base.pt not found at $PXR_REPO/models/kermt/grover_base.pt. Download via gdown (see models/kermt/README.md)." >&2
    exit 1
fi

cd "$KERMT_REPO"
export PYTHONPATH="$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "KERMT pretrain: seed=$SEED, save_dir=$SAVE_DIR"
pixi run --manifest-path "$KERMT_REPO/pixi.toml" python main.py finetune \
    --data_path "$PXR_REPO/data/kermt/pretrain_train.csv" \
    --separate_val_path "$PXR_REPO/data/kermt/pretrain_val.csv" \
    --save_dir "$SAVE_DIR" \
    --checkpoint_path "$PXR_REPO/models/kermt/grover_base.pt" \
    --dataset_type regression \
    --split_type scaffold_balanced \
    --ensemble_size 1 \
    --num_folds 1 \
    --no_features_scaling \
    --ffn_hidden_size 256 \
    --ffn_num_layers 3 \
    --bond_drop_rate 0.1 \
    --epochs "$EPOCHS" \
    --metric mae \
    --self_attention \
    --dist_coff 0.15 \
    --max_lr 1e-4 \
    --final_lr 2e-5 \
    --dropout 0.1 \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"
