"""Average Uni-Mol v2 log2fc embeddings across 5 seeds (42, 43, 44, 45, 46).

Per Codex 2026-05-02 priority #1: variance-reduction ensemble. Same encoder,
different seeds → per-row mean. Memory `reference_multi_seed_pretrain_recipe`
established this recipe drove rank 1 with 5-seed chemprop in PR #120.

Inputs:
  data/unimol_v2_log2fc_real_embed.parquet           (seed=42)
  data/unimol_v2_log2fc_seed{43,44,45,46}_embed.parquet

Output:
  data/unimol_v2_log2fc_seed5ens_embed.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/nagaet/pxr-iduction-challenge")

INPUTS = [
    REPO.joinpath("data", "unimol_v2_log2fc_real_embed.parquet"),       # seed 42
    REPO.joinpath("data", "unimol_v2_log2fc_seed43_embed.parquet"),
    REPO.joinpath("data", "unimol_v2_log2fc_seed44_embed.parquet"),
    REPO.joinpath("data", "unimol_v2_log2fc_seed45_embed.parquet"),
    REPO.joinpath("data", "unimol_v2_log2fc_seed46_embed.parquet"),
]
OUT = REPO.joinpath("data", "unimol_v2_log2fc_seed5ens_embed.parquet")


def main() -> None:
    dfs = []
    for p in INPUTS:
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        df = pd.read_parquet(p)
        dfs.append(df)
        print(f"  {p.name}: shape={df.shape}, idx={df.index.name}")

    ref_idx = dfs[0].index
    for i, d in enumerate(dfs[1:], start=1):
        if not d.index.equals(ref_idx):
            d_aligned = d.reindex(ref_idx)
            if d_aligned.isna().any().any():
                raise SystemExit(f"Index mismatch at input {i}; cannot reindex cleanly")
            dfs[i] = d_aligned
            print(f"  reindexed input {i}")

    arr = np.stack([d.to_numpy(dtype=np.float32) for d in dfs], axis=0)
    print(f"  stacked: {arr.shape} (seeds, rows, dims)")

    avg = arr.mean(axis=0).astype(np.float32)
    print(f"  averaged: {avg.shape}")

    print(f"  inter-seed std (mean, median, p95): "
          f"{arr.std(axis=0).mean():.4f}, "
          f"{np.median(arr.std(axis=0)):.4f}, "
          f"{np.percentile(arr.std(axis=0), 95):.4f}")

    out_df = pd.DataFrame(avg, index=ref_idx, columns=dfs[0].columns)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT)
    print(f"\nWrote {OUT} shape {out_df.shape}")


if __name__ == "__main__":
    main()
