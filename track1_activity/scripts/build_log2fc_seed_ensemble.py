"""Average per-seed chemprop log2fc_pred parquets into a 5-seed ensemble.

Multi-seed pretrain (Plan A, 2026-04-25): same chemprop architecture
+ hyperparams as the production single-seed pretrain, but trained 5
independent times with seeds [42, 43, 44, 45, 46]. Averaging per-row
predictions reduces variance without changing bias, and is the
canonical deep-learning ensembling trick.

This is *unlike* the prior 4-encoder log2fc ensemble (PR #116) which
mixed different architectures — that diluted chemprop's strong signal
because the other encoders were weaker. Same-arch multi-seed only
averages noise, so the ensemble should be at least as good as the
single seed (and typically -0.001 to -0.005 log2fc MAE better).

Pipeline:
  1. Generate per-seed parquets first by running:
       pixi run python track1_activity/scripts/run_chemprop_predict_log2fc.py \\
         --ckpt track1_activity/checkpoints/chemprop_pretrain_seed{N}/pretrain.pt \\
         --out  data/chemprop_pretrain_log2fc_predictions_seed{N}.parquet
     for N in {43, 44, 45, 46}. Seed 42 already lives at the default
     path data/chemprop_pretrain_log2fc_predictions.parquet.
  2. Run this script: averages all 5 column-wise, writes
     data/chemprop_pretrain_log2fc_predictions_seed5ens.parquet.
  3. Add a feature handler `cheme_2d_full_boltz_log2fc_pred_seed5ens`
     in run_train.py and run TabPFN UMAP CV to compare to baseline.

Usage:
    pixi run python track1_activity/scripts/build_log2fc_seed_ensemble.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.joinpath("data")

SEEDS = (42, 43, 44, 45, 46)


def parquet_path_for_seed(seed: int) -> Path:
    if seed == 42:
        return DATA_DIR.joinpath("chemprop_pretrain_log2fc_predictions.parquet")
    return DATA_DIR.joinpath(f"chemprop_pretrain_log2fc_predictions_seed{seed}.parquet")


def main() -> None:
    print(f"Loading per-seed parquets for seeds {SEEDS}")
    frames: list[pd.DataFrame] = []
    for s in SEEDS:
        p = parquet_path_for_seed(s)
        if not p.exists():
            raise SystemExit(
                f"Missing seed {s} parquet: {p}\n"
                "Run run_chemprop_predict_log2fc.py for that seed first."
            )
        df = pd.read_parquet(p)
        df.index = df.index.astype(int)
        frames.append(df.sort_index())
        print(
            f"  seed={s}: shape={df.shape} "
            f"log2fc_8p25 mean={df['log2fc_8p25_pred'].mean():.4f} "
            f"std={df['log2fc_8p25_pred'].std():.4f}"
        )

    # Sanity: same row index across all 5
    base_idx = frames[0].index
    for s, df in zip(SEEDS[1:], frames[1:], strict=True):
        if not df.index.equals(base_idx):
            raise SystemExit(
                f"seed {s} parquet index does not match seed 42; "
                "re-run with same compound list."
            )

    # Per-row mean over seeds (numpy stack to keep contiguous block)
    cols = ["log2fc_8p25_pred", "log2fc_33_pred"]
    stacked = np.stack(
        [df[cols].to_numpy(dtype=np.float64) for df in frames], axis=0
    )  # (n_seeds, n_compounds, 2)
    mean = stacked.mean(axis=0)
    std_across_seeds = stacked.std(axis=0)

    # Build output df
    out = pd.DataFrame(mean, index=base_idx, columns=cols)
    out.index.name = "compound_id"
    out_path = DATA_DIR.joinpath(
        "chemprop_pretrain_log2fc_predictions_seed5ens.parquet"
    )
    out.to_parquet(out_path)
    print(f"\nWrote {out_path} ({len(out)} rows)")
    print(out.describe())

    # Inter-seed agreement diagnostic
    print(
        f"\nInter-seed std (per row, averaged over 13136 compounds):"
        f"\n  log2fc_8p25_pred: {std_across_seeds[:, 0].mean():.4f}"
        f"\n  log2fc_33_pred:   {std_across_seeds[:, 1].mean():.4f}"
    )


if __name__ == "__main__":
    main()
