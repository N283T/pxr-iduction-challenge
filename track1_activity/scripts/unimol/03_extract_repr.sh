#!/usr/bin/env bash
# Extract Uni-Mol v2 molecule-level representation (concat: CLS + mean-pool +
# max-pool of atomic reprs = 3 x 768 = 2304d) for all 13,136 compounds.
#
# Output: data/unimol/cls_repr.npz with keys:
#   - compound_id (int64, N)
#   - cls_repr    (float32, N x 2304)  -- concat [cls, mean_atomic, max_atomic]
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
CKPT_DIR="${CKPT_DIR:-$PXR_REPO/models/unimol_v2_log2fc}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
INPUT_CSV="${INPUT_CSV:-$PXR_REPO/data/unimol/pretrain_all.csv}"
OUT_NPZ="${OUT_NPZ:-$PXR_REPO/data/unimol/cls_repr.npz}"

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "ERROR: $INPUT_CSV not found" >&2
    exit 1
fi

export UNIMOL_WEIGHT_DIR="$CKPT_DIR"

cat > /tmp/unimol_extract_repr.py <<PYEOF
import argparse
import numpy as np
import pandas as pd
from unimol_tools import UniMolRepr

ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
args = ap.parse_args()

df = pd.read_csv(args.csv)
assert 'SMILES' in df.columns
smiles = df['SMILES'].tolist()
cids = df['compound_id'].astype(int).tolist()
print(f'Loaded {len(smiles)} SMILES')

r = UniMolRepr(model_name='unimolv2', model_size=args.model_size, use_cuda=True)
out = r.get_repr(data=smiles, return_atomic_reprs=True)

cls = np.array(out['cls_repr'], dtype=np.float32)
atomic = out['atomic_reprs']  # list of (n_atoms_i, 768) per molecule

# Compute mean and max pool per molecule
n = len(smiles)
emb_dim = cls.shape[1]
mean_pool = np.zeros((n, emb_dim), dtype=np.float32)
max_pool = np.zeros((n, emb_dim), dtype=np.float32)
for i, a in enumerate(atomic):
    arr = np.asarray(a, dtype=np.float32)
    if arr.ndim == 1 or arr.shape[0] == 0:
        # fallback: use cls
        mean_pool[i] = cls[i]
        max_pool[i] = cls[i]
    else:
        mean_pool[i] = arr.mean(axis=0)
        max_pool[i] = arr.max(axis=0)

concat = np.concatenate([cls, mean_pool, max_pool], axis=1)
print(f'cls {cls.shape}  mean {mean_pool.shape}  max {max_pool.shape}  concat {concat.shape}')
np.savez(args.out, compound_id=np.array(cids, dtype=np.int64), cls_repr=concat)
print(f'saved: {args.out}')
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_extract_repr.py \
    --csv "$INPUT_CSV" \
    --out "$OUT_NPZ" \
    --model_size "$MODEL_SIZE"
