#!/usr/bin/env bash
# Extract Uni-Mol v2 CLS representation for all 13,136 compounds using the
# log2_fc-finetuned checkpoint from Task 3.
#
# Output: data/unimol/cls_repr.npz (keys: compound_id, cls_repr)
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

# Point unimol_tools UniMolRepr at our finetuned checkpoint dir
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
print(f'Loaded {len(smiles)} SMILES for extraction')

r = UniMolRepr(model_name='unimolv2', model_size=args.model_size, use_cuda=True)
out = r.get_repr(data=smiles)
cls = np.array(out['cls_repr'], dtype=np.float32)
print(f'cls shape: {cls.shape}')
np.savez(args.out, compound_id=np.array(cids, dtype=np.int64), cls_repr=cls)
print(f'saved: {args.out}')
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_extract_repr.py \
    --csv "$INPUT_CSV" \
    --out "$OUT_NPZ" \
    --model_size "$MODEL_SIZE"
