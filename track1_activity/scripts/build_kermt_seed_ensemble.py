"""Average per-seed kermt embedding parquets into an N-seed ensemble.

Mirrors `build_log2fc_seed_ensemble.py` but for the kermt
graph-transformer pretrain (which uses 3200d frozen embeddings rather
than 2 predicted log2_fc scalars).

Same-arch (GROVER-base + LoRA-style continued pretrain on log2_fc)
multi-seed averaging is sound because the encoder weights only differ
in random-init / dropout patterns; the embedding space is aligned.

Pipeline:
  1. For each seed N: produce data/kermt_pretrain_embed_seed{N}.parquet
     by chaining `run_kermt_embed_extract.sh` (with SEED=N + custom
     OUTPUT_NPZ) and `kermt_embed_npz_to_parquet.py --npz ... --out ...`.
     Seed 0 lives at the canonical path data/kermt_pretrain_embed.parquet
     (KERMT default = original production checkpoint).
  2. Run this script with --seeds 0 43 44 45 46 (or arbitrary set) to
     row-mean the parquets. Output:
       data/kermt_pretrain_embed_seed{N}ens.parquet
     where N = number of seeds.

Usage:
    pixi run python track1_activity/scripts/build_kermt_seed_ensemble.py \\
        --seeds 0 43 44 45 46
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.joinpath("data")


def parquet_path_for_seed(seed: int) -> Path:
    if seed == 0:
        return DATA_DIR.joinpath("kermt_pretrain_embed.parquet")
    return DATA_DIR.joinpath(f"kermt_pretrain_embed_seed{seed}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seeds to average (e.g. --seeds 0 43 44 45 46)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet. Defaults to "
        "data/kermt_pretrain_embed_seed{N}ens.parquet where N = len(seeds).",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    out_path = args.out or DATA_DIR.joinpath(
        f"kermt_pretrain_embed_seed{len(seeds)}ens.parquet"
    )

    print(f"Loading per-seed parquets for seeds {seeds}")
    frames: list[pd.DataFrame] = []
    for s in seeds:
        p = parquet_path_for_seed(s)
        if not p.exists():
            raise SystemExit(
                f"Missing seed {s} parquet: {p}\n"
                "Run kermt_embed_npz_to_parquet.py with --npz/--out for that seed."
            )
        df = pd.read_parquet(p)
        df.index = df.index.astype(int)
        frames.append(df.sort_index())
        print(
            f"  seed={s}: shape={df.shape} mean={df.values.mean():.4f} "
            f"std={df.values.std():.4f}"
        )

    base_idx = frames[0].index
    base_cols = list(frames[0].columns)
    for s, df in zip(seeds[1:], frames[1:], strict=True):
        if not df.index.equals(base_idx):
            raise SystemExit(
                f"seed {s} parquet index does not match seed {seeds[0]}; "
                "re-run with same compound list."
            )
        if list(df.columns) != base_cols:
            raise SystemExit(f"seed {s} columns do not match seed {seeds[0]}.")

    stacked = np.stack(
        [df[base_cols].to_numpy(dtype=np.float64) for df in frames], axis=0
    )
    mean = stacked.mean(axis=0)
    std_across_seeds = stacked.std(axis=0)

    out = pd.DataFrame(mean, index=base_idx, columns=base_cols)
    out.index.name = "compound_id"
    out.to_parquet(out_path)
    print(f"\nWrote {out_path} ({len(out)} rows, {len(base_cols)} dims)")
    print(
        f"Inter-seed std (per row, averaged over {len(out)} compounds × "
        f"{len(base_cols)} dims): {std_across_seeds.mean():.4f}"
    )


if __name__ == "__main__":
    main()
