#!/usr/bin/env bash
# Extract Uni-Mol v2 repr (CLS+atom-mean+atom-max concat 2304d) using a
# CUSTOM finetuned checkpoint, FORCING load_pretrained_weights() after
# UniMolRepr() construction (which by default loads public HF weights).
#
# This fixes the silent fallback bug noted in PR #114 closeout.
#
# Required env vars:
#   CKPT_PATH: path to model_0.pth (or any .pth from MolTrain output)
#   INPUT_CSV: SMILES list (must have 'SMILES' column + 'compound_id')
#   OUT_NPZ:   output npz path (compound_id + cls_repr 2304d)
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"
CKPT_PATH="${CKPT_PATH:?ERROR: set CKPT_PATH to a finetuned model_X.pth}"
INPUT_CSV="${INPUT_CSV:-$PXR_REPO/data/unimol/pretrain_all.csv}"
OUT_NPZ="${OUT_NPZ:?ERROR: set OUT_NPZ to output path}"

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: CKPT_PATH=$CKPT_PATH does not exist" >&2
    exit 1
fi
if [[ ! -f "$INPUT_CSV" ]]; then
    echo "ERROR: INPUT_CSV=$INPUT_CSV does not exist" >&2
    exit 1
fi

cat > /tmp/unimol_extract_repr_v2.py <<PYEOF
import argparse, numpy as np, pandas as pd, torch
from unimol_tools import UniMolRepr

ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
args = ap.parse_args()

df = pd.read_csv(args.csv)
assert 'SMILES' in df.columns
smiles = df['SMILES'].tolist()
cids = df['compound_id'].astype(int).tolist()
print(f'Loaded {len(smiles)} SMILES')

r = UniMolRepr(model_name='unimolv2', model_size=args.model_size, use_cuda=True)
print(f'UniMolRepr ready (default public weights loaded)')

# Override default public weights with our finetuned checkpoint
r.model.load_pretrained_weights(path=args.ckpt, strict=False)
print(f'Reloaded weights from {args.ckpt}')
r.model.eval()

out = r.get_repr(data=smiles, return_atomic_reprs=True)

cls = np.array(out['cls_repr'], dtype=np.float32)
atomic = out['atomic_reprs']

n = len(smiles)
emb_dim = cls.shape[1]
mean_pool = np.zeros((n, emb_dim), dtype=np.float32)
max_pool = np.zeros((n, emb_dim), dtype=np.float32)
for i, atoms in enumerate(atomic):
    arr = np.asarray(atoms, dtype=np.float32)
    if arr.size == 0:
        continue
    mean_pool[i] = arr.mean(axis=0)
    max_pool[i] = arr.max(axis=0)

concat = np.concatenate([cls, mean_pool, max_pool], axis=1)
print(f'cls {cls.shape}  mean {mean_pool.shape}  max {max_pool.shape}  concat {concat.shape}')

np.savez(args.out, compound_id=np.array(cids, dtype=np.int64), cls_repr=concat)
print(f'saved: {args.out}')
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_extract_repr_v2.py \
    --csv "$INPUT_CSV" \
    --ckpt "$CKPT_PATH" \
    --out "$OUT_NPZ" \
    --model_size "$MODEL_SIZE"
