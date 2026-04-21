#!/bin/bash
# Extract KERMT (GROVER) fingerprint (pre-FFN pooled graph representation)
# for all compounds in pretrain_all.csv, using the continued-pretrain
# checkpoint produced by Task 5 (run_kermt_pretrain.sh).
#
# Output: data/kermt/embeddings.npz (KERMT native format)
#
# KERMT's main.py fingerprint CLI flags (see `main.py fingerprint --help`):
#   --data_path          Input CSV where the first column is SMILES
#   --output_path        Path to npz file where predictions will be saved
#   --checkpoint_path    Path to model.pt (continued-pretrain output)
#   --fingerprint_source {atom,bond,both}
#   --batch_size         Batch size (default 32)
# Notes:
#   - The fingerprint subcommand does NOT accept --no_features_scaling;
#     scaling toggles are only for the finetune subcommand.
#   - KERMT reads SMILES from the FIRST CSV column (not a named `smiles`
#     column) and treats remaining columns as numeric targets. Our
#     canonical pretrain_all.csv has [compound_id, smiles], so we write a
#     SMILES-only CSV for KERMT and re-attach compound_ids after.
#   - KERMT 0.x task/fingerprint.py has a bug: load_checkpoint returns
#     (model, state) but the task forgets to unpack it and then calls
#     .eval() on the tuple. We route through
#     kermt_fingerprint_patched.py which replicates the fingerprint branch
#     of main.py with the tuple unpacking fixed, sharing all other logic
#     (MolVocab init, MoleculeDataset, MolCollator, do_generate loop) via
#     the KERMT package import.
#
# Env vars:
#   LIMIT=100            Smoke test on first 100 compounds (default 0 = all).
#   BATCH_SIZE=32        Override batch size.
#   KERMT_REPO=<path>    Override KERMT repo location.
#   PXR_REPO=<path>      Override PXR repo location.
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
KERMT_REPO="${KERMT_REPO:-$HOME/ghq/github.com/NVIDIA-Digital-Bio/KERMT}"
LIMIT="${LIMIT:-0}"  # 0 = all compounds; positive = smoke test
BATCH_SIZE="${BATCH_SIZE:-32}"

# Pre-flight
if [[ ! -f "$KERMT_REPO/main.py" ]]; then
    echo "ERROR: KERMT repo not found at $KERMT_REPO (override with KERMT_REPO=...)" >&2
    exit 1
fi
if [[ ! -f "$PXR_REPO/data/kermt/pretrain_all.csv" ]]; then
    echo "ERROR: pretrain_all.csv not found. Run prepare_kermt_pretrain_csv.py first." >&2
    exit 1
fi
if [[ ! -f "$PXR_REPO/models/kermt/pretrain/fold_0/model_0/model.pt" ]]; then
    echo "ERROR: Pretrained checkpoint not found at $PXR_REPO/models/kermt/pretrain/fold_0/model_0/model.pt. Run run_kermt_pretrain.sh first." >&2
    exit 1
fi

mkdir -p "$PXR_REPO/data/kermt"

# KERMT expects the CSV's first column to be SMILES (treats remaining columns as
# numeric targets). Our pretrain_all.csv is [compound_id, smiles], so write a
# SMILES-only CSV for the fingerprint call. The original CSV is still the
# canonical id/order source for the npz->parquet converter.
SMILES_CSV="$PXR_REPO/data/kermt/pretrain_all_smiles.csv"
awk -F',' 'BEGIN{print "smiles"} NR>1{print $2}' \
    "$PXR_REPO/data/kermt/pretrain_all.csv" > "$SMILES_CSV"

INPUT_CSV="$SMILES_CSV"
if [[ "$LIMIT" -gt 0 ]]; then
    TMP_CSV="$PXR_REPO/data/kermt/pretrain_all_smiles_head${LIMIT}.csv"
    head -n $((LIMIT + 1)) "$SMILES_CSV" > "$TMP_CSV"
    INPUT_CSV="$TMP_CSV"
    echo "Smoke test mode: using $INPUT_CSV (first $LIMIT compounds)"
fi

cd "$KERMT_REPO"
export PYTHONPATH="$PWD"
# KERMT enables torch.use_deterministic_algorithms, which requires this
# env var for CuBLAS linear kernels under CUDA >= 10.2.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

pixi run --manifest-path "$KERMT_REPO/pixi.toml" python \
    "$PXR_REPO/track1_activity/scripts/kermt_fingerprint_patched.py" fingerprint \
    --data_path "$INPUT_CSV" \
    --checkpoint_path "$PXR_REPO/models/kermt/pretrain/fold_0/model_0/model.pt" \
    --fingerprint_source both \
    --output_path "$PXR_REPO/data/kermt/embeddings.npz" \
    --batch_size "$BATCH_SIZE"
