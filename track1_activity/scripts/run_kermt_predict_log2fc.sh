#!/bin/bash
# Predict log2_fc @ 8.25uM / 33uM for 4653 train+test compounds using
# the KERMT (GROVER + D-MPNN) continued-pretrain checkpoint.
#
# Phase 1 of issue #115 (log2fc_pred ensembling). Mirrors
# run_chemprop_predict_log2fc.py but for KERMT. Invokes KERMT's `predict`
# subcommand from its own pixi env (KERMT has torch 2.2 pinned; main
# pixi env has torch 2.6). Post-processes the SMILES-indexed output CSV
# back into a compound_id-indexed parquet matching the shape of
# data/chemprop_pretrain_log2fc_predictions.parquet.
#
# Inputs:
#   - Pretrain checkpoint: models/kermt/pretrain/fold_0/model_0/model.pt
#   - Train+test SMILES CSV: generated inline (4653 rows)
#
# Output:
#   data/kermt_pretrain_log2fc_predictions.parquet
#     index=compound_id, columns=[log2fc_8p25_pred, log2fc_33_pred]
#
# Usage:
#   bash track1_activity/scripts/run_kermt_predict_log2fc.sh
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
KERMT_REPO="${KERMT_REPO:-$HOME/ghq/github.com/NVIDIA-Digital-Bio/KERMT}"
BATCH_SIZE="${BATCH_SIZE:-32}"

CHECKPOINT_DIR="$PXR_REPO/models/kermt/pretrain/fold_0"
CHECKPOINT_FILE="$CHECKPOINT_DIR/model_0/model.pt"
WORK_DIR="$PXR_REPO/data/kermt"
TRAIN_TEST_CSV="$WORK_DIR/train_test_smiles.csv"
PRED_CSV_RAW="$WORK_DIR/train_test_log2fc_predictions.csv"
OUT_PARQUET="$PXR_REPO/data/kermt_pretrain_log2fc_predictions.parquet"

# Pre-flight
if [[ ! -f "$KERMT_REPO/main.py" ]]; then
    echo "ERROR: KERMT repo not found at $KERMT_REPO (override with KERMT_REPO=...)" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
    echo "ERROR: KERMT pretrain checkpoint not found at $CHECKPOINT_FILE" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"

# Step 1: export train+test SMILES (4653 rows) from DB. KERMT expects the
# first CSV column to be SMILES; we emit [compound_id, smiles] then strip
# compound_id into a separate index CSV before invoking predict.
echo "Exporting train+test SMILES to $TRAIN_TEST_CSV"
cd "$PXR_REPO"
pixi run python -c "
import pandas as pd, psycopg2, sys
sys.path.insert(0, '$PXR_REPO/track1_activity/src')
from data import DB_PARAMS
sql = '''
SELECT DISTINCT c.id AS compound_id, c.std_smiles AS smiles
FROM compounds c
WHERE c.id IN (
  SELECT compound_id FROM train_activity
  UNION
  SELECT compound_id FROM test_activity
)
  AND c.std_smiles IS NOT NULL
ORDER BY c.id
'''
with psycopg2.connect(**DB_PARAMS) as conn:
    df = pd.read_sql(sql, conn)
df.to_csv('$TRAIN_TEST_CSV', index=False)
print(f'  exported {len(df)} rows')
df[['smiles']].to_csv('$WORK_DIR/train_test_smiles_only.csv', index=False)
"

# Step 2: invoke KERMT's predict in its own pixi env. KERMT expects
# checkpoint_dir (walks to find *.pt). Point at fold_0 so it picks up
# the single model_0/model.pt under it.
cd "$KERMT_REPO"
export PYTHONPATH="$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Running KERMT predict (checkpoint_dir=$CHECKPOINT_DIR)"
pixi run --manifest-path "$KERMT_REPO/pixi.toml" python main.py predict \
    --data_path "$WORK_DIR/train_test_smiles_only.csv" \
    --output_path "$PRED_CSV_RAW" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --no_features_scaling

# Step 3: post-process to compound_id-indexed parquet
echo "Post-processing predictions -> $OUT_PARQUET"
cd "$PXR_REPO"
pixi run python -c "
import pandas as pd
idx = pd.read_csv('$TRAIN_TEST_CSV')
raw = pd.read_csv('$PRED_CSV_RAW')
# KERMT predict CSV indexes by SMILES; align by row order (we passed them
# in compound_id order and KERMT preserves order).
assert len(idx) == len(raw), f'row count mismatch: idx={len(idx)} raw={len(raw)}'
# First column of raw is the smiles index
smiles_col = raw.columns[0]
if (idx['smiles'].values != raw[smiles_col].values).any():
    n_mismatch = int((idx['smiles'].values != raw[smiles_col].values).sum())
    raise ValueError(f'SMILES order mismatch between input and KERMT output: {n_mismatch} rows differ')
# Identify the 2 target columns (task_names from pretrain: log2fc_8p25, log2fc_33)
target_cols = [c for c in raw.columns if c != smiles_col]
assert len(target_cols) == 2, f'expected 2 targets, got {target_cols}'
# Match chemprop output schema
out = pd.DataFrame({
    'compound_id': idx['compound_id'].values,
    'log2fc_8p25_pred': raw[target_cols[0]].astype(float).values,
    'log2fc_33_pred': raw[target_cols[1]].astype(float).values,
}).set_index('compound_id')
out.to_parquet('$OUT_PARQUET')
print(f'Saved {out.shape} to $OUT_PARQUET')
print(f'  log2fc_8p25_pred: mean={out.log2fc_8p25_pred.mean():.3f} std={out.log2fc_8p25_pred.std():.3f}')
print(f'  log2fc_33_pred:   mean={out.log2fc_33_pred.mean():.3f} std={out.log2fc_33_pred.std():.3f}')
"
