#!/usr/bin/env -S pixi run python
"""Predict log2_fc @ 8.25uM / 33uM for all train+test compounds.

Runs the full chemprop pretrain MPNN (encoder + predictor) forward on
train+test SMILES to produce predictions for both pretrain heads.
Outputs z-scored and un-z-scored values; downstream feature bundles
typically use un-z-scored (the scaling is irrelevant for TabPFN/LGBM).

Buterez 2024 strategy-2 pure: predicted LF labels as side features for
the HF regressor. Chosen over strategy-4 "hybrid" (real train /
predicted test) because our test compounds have 0 overlap with
single_concentration, so a hybrid would leak a train/test distribution
mismatch.

Output: data/chemprop_pretrain_log2fc_predictions.parquet indexed by
compound_id with columns log2fc_8p25_pred, log2fc_33_pred (un-z-scored).

Usage:
    pixi run python track1_activity/scripts/run_chemprop_predict_log2fc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from chemprop import data as chemprop_data  # noqa: E402

# Reuse the build function from the embed extractor to keep arch identical
from run_chemprop_embed_extract import (  # noqa: E402
    build_pretrain_model,
    load_target_compounds,
)

from data import DB_PARAMS  # noqa: E402

CKPT_PATH = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemprop_pretrain", "pretrain.pt"
)
OUT_PATH = REPO_ROOT.joinpath("data", "chemprop_pretrain_log2fc_predictions.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=CKPT_PATH,
        help=f"Path to chemprop pretrain .pt (default: {CKPT_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=f"Output parquet path (default: {OUT_PATH})",
    )
    parser.add_argument(
        "--scope",
        choices=["train_test", "htchem", "all_with_smiles"],
        default="train_test",
        help="compound set to predict",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"missing pretrain ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    means = np.asarray(ckpt["target_means"], dtype=np.float32)  # (2,)
    stds = np.asarray(ckpt["target_stds"], dtype=np.float32)  # (2,)
    print(f"Target standardisation from pretrain meta: mean={means}, std={stds}")

    df = load_target_compounds(args.scope)
    n = len(df)
    print(f"Target compounds ({args.scope}): {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pretrain_model(params).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model on {device}, loaded state_dict")

    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.full(2, 0.0, dtype=np.float32))
        for smi in df["smiles"]
    ]
    dataset = chemprop_data.MoleculeDataset(pts)
    loader = chemprop_data.build_dataloader(dataset, batch_size=256, shuffle=False)

    preds_z: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            # Full forward returns z-scored (2,) since pretrain targets were standardised
            preds = model(bmg, V_d, X_d).detach().cpu().numpy()  # (B, 2)
            preds_z.append(preds)

    preds_z_arr = np.concatenate(preds_z, axis=0)  # (n, 2)
    assert preds_z_arr.shape == (n, 2)

    # Un-z-score using the pretrain-saved stats
    preds_raw = preds_z_arr * stds + means

    out = pd.DataFrame(
        {
            "compound_id": df["compound_id"].values,
            "log2fc_8p25_pred": preds_raw[:, 0],
            "log2fc_33_pred": preds_raw[:, 1],
        }
    ).set_index("compound_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)

    print(f"Saved {out.shape} to {args.out}")
    print(
        "  log2fc_8p25_pred: mean=%.3f std=%.3f"
        % (out.log2fc_8p25_pred.mean(), out.log2fc_8p25_pred.std())
    )
    print(
        "  log2fc_33_pred:   mean=%.3f std=%.3f"
        % (out.log2fc_33_pred.mean(), out.log2fc_33_pred.std())
    )

    # Sanity check: compare predicted vs real for train compounds that
    # have single_conc data, to measure LF model quality on train.
    sql = """
    SELECT compound_id,
      AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
               THEN log2_fc_estimate END) AS log2fc_8p25,
      AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
               THEN log2_fc_estimate END) AS log2fc_33
    FROM single_concentration
    GROUP BY compound_id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        real_df = pd.read_sql(sql, conn).set_index("compound_id")
    joined = out.join(real_df, how="inner")
    m8 = joined.log2fc_8p25.notna()
    m33 = joined.log2fc_33.notna()
    if m8.sum() > 10:
        from scipy.stats import pearsonr, spearmanr

        r8 = pearsonr(
            joined.loc[m8, "log2fc_8p25_pred"], joined.loc[m8, "log2fc_8p25"]
        ).statistic
        sp8 = spearmanr(
            joined.loc[m8, "log2fc_8p25_pred"], joined.loc[m8, "log2fc_8p25"]
        ).correlation
        mae8 = float(
            np.mean(
                np.abs(
                    joined.loc[m8, "log2fc_8p25_pred"] - joined.loc[m8, "log2fc_8p25"]
                )
            )
        )
        print(
            f"\n  LF quality @ 8.25uM on {int(m8.sum())} overlapping compounds: "
            f"r={r8:.3f}, spearman={sp8:.3f}, MAE={mae8:.3f}"
        )
    if m33.sum() > 10:
        from scipy.stats import pearsonr, spearmanr

        r33 = pearsonr(
            joined.loc[m33, "log2fc_33_pred"], joined.loc[m33, "log2fc_33"]
        ).statistic
        sp33 = spearmanr(
            joined.loc[m33, "log2fc_33_pred"], joined.loc[m33, "log2fc_33"]
        ).correlation
        mae33 = float(
            np.mean(
                np.abs(joined.loc[m33, "log2fc_33_pred"] - joined.loc[m33, "log2fc_33"])
            )
        )
        print(
            f"  LF quality @ 33uM on {int(m33.sum())} overlapping compounds: "
            f"r={r33:.3f}, spearman={sp33:.3f}, MAE={mae33:.3f}"
        )


if __name__ == "__main__":
    main()
