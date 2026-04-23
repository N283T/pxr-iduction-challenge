"""Convert Uni-Mol CLS repr npz -> parquet for run_train.py consumption.

Input:  data/unimol/cls_repr.npz (keys: compound_id, cls_repr)
Output: data/unimol_v2_pretrain_embed.parquet (index=compound_id,
        columns=emb_0000..emb_NNNN)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_PATH = REPO_ROOT.joinpath("data", "unimol", "cls_repr.npz")
OUT_PATH = REPO_ROOT.joinpath("data", "unimol_v2_pretrain_embed.parquet")


def main() -> None:
    data = np.load(IN_PATH)
    cids = data["compound_id"].astype(int)
    cls = data["cls_repr"].astype(np.float32)
    print(f"Loaded npz: compound_id {cids.shape} cls_repr {cls.shape}")
    if cids.shape[0] != cls.shape[0]:
        raise SystemExit("row count mismatch")

    cols = [f"emb_{i:04d}" for i in range(cls.shape[1])]
    df = pd.DataFrame(cls, index=pd.Index(cids, name="compound_id"), columns=cols)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
