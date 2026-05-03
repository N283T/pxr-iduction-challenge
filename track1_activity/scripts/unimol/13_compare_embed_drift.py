"""Compare two Uni-Mol v2 embedding parquets (representation drift).

Used in Phase A decision rule (Codex 2026-05-02): if the filtered-pretrain
embed differs little from the current 0.4885 baseline, the FT had no
representational effect and Uni-Mol should be killed regardless of OOF MAE.

Usage:
    pixi run python 13_compare_embed_drift.py \
        --baseline data/unimol_v2_log2fc_real_embed.parquet \
        --candidate data/unimol_v2_log2fc_filtered_embed.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    base = pd.read_parquet(args.baseline)
    cand = pd.read_parquet(args.candidate)
    common = base.index.intersection(cand.index)
    print(f"baseline: {base.shape}  candidate: {cand.shape}  common: {len(common)}")

    a = base.loc[common].to_numpy(dtype=np.float32)
    b = cand.loc[common].to_numpy(dtype=np.float32)

    diff = a - b
    abs_diff = np.abs(diff)

    # Per-row L2 distance and cosine similarity
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    cos = (a * b).sum(axis=1) / (a_norm.squeeze() * b_norm.squeeze() + 1e-12)
    l2_per_row = np.linalg.norm(diff, axis=1)

    rel_l2 = l2_per_row / (a_norm.squeeze() + 1e-12)

    print("\n=== Embedding drift summary ===")
    print(f"  abs diff:    max {abs_diff.max():.4f}  mean {abs_diff.mean():.4f}")
    print(
        f"  per-row L2:  max {l2_per_row.max():.4f}  mean {l2_per_row.mean():.4f}  "
        f"median {np.median(l2_per_row):.4f}"
    )
    print(
        f"  relative L2: max {rel_l2.max():.4f}  mean {rel_l2.mean():.4f}  "
        f"median {np.median(rel_l2):.4f}"
    )
    print(
        f"  cosine sim:  min {cos.min():.4f}  mean {cos.mean():.4f}  "
        f"median {np.median(cos):.4f}"
    )

    # Verdict heuristics
    if cos.mean() > 0.999:
        verdict = "NO DRIFT (cos>0.999) — representations are nearly identical"
    elif cos.mean() > 0.99:
        verdict = "TINY DRIFT (cos>0.99) — small movement"
    elif cos.mean() > 0.95:
        verdict = "MODERATE DRIFT (cos>0.95) — some real change"
    else:
        verdict = "LARGE DRIFT (cos<0.95) — substantial representational shift"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
