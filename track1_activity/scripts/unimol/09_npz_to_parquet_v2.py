"""Convert Uni-Mol CLS repr npz -> parquet (multi-variant support).

Same transform as 04_npz_to_parquet.py but with --in / --out CLI args so
we can target any variant checkpoint (multitask, pec50_ft, multi-seed, etc.).

Usage:
    pixi run python 09_npz_to_parquet_v2.py \
        --in  data/unimol/cls_repr_multitask.npz \
        --out data/unimol_v2_multitask_embed.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    data = np.load(in_path)
    cids = data["compound_id"].astype(int)
    cls = data["cls_repr"].astype(np.float32)
    print(f"Loaded {in_path}: compound_id {cids.shape} cls_repr {cls.shape}")
    if cids.shape[0] != cls.shape[0]:
        raise SystemExit("row count mismatch")

    cols = [f"emb_{i:04d}" for i in range(cls.shape[1])]
    df = pd.DataFrame(cls, index=pd.Index(cids, name="compound_id"), columns=cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Wrote {out_path} shape {df.shape}")


if __name__ == "__main__":
    main()
