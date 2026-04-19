"""Diversity analysis for GatorAffinity zero-shot predictions.

Compares preds_zero_shot_ft.csv against:
  - raw pEC50 (Pearson, Spearman)
  - 8-model ensemble members' OOF predictions (pairwise Pearson)
  - linearly-calibrated MAE (shift + scale to pEC50)

Takeaway: the lower the cross-correlation to existing members AND
the competitive the calibrated MAE, the more ensemble value Gator brings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import DB_PARAMS  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402


POOL = (
    "chemprop_optuna_umap",
    "chemprop_chemeleon_umap",
    "chemprop_multitask5_umap_aux0.0_tuned",
    "attentivefp_optuna_umap",
    "gatedgcn_optuna_umap",
    "residual_physprop+mordred_umap",
    "tabpfn_2d_full_boltz_umap",
    "tabpfn_chemeleon_umap",
)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rae(y, p):
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def main() -> None:
    csv_path = REPO_ROOT.joinpath("structures", "gator", "preds_zero_shot_ft.csv")
    df = pd.read_csv(csv_path)
    # PDB_ID format is "{compound_id:05d}_UNL" from process_pdbs.py.
    df["compound_id"] = df["PDB_ID"].str.split("_").str[0].astype(int)
    df = df.sort_values("compound_id").reset_index(drop=True)

    conn = psycopg2.connect(**DB_PARAMS)
    y_df = pd.read_sql(
        "SELECT compound_id, pec50 FROM train_activity ORDER BY id", conn
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (list(POOL),),
    )
    name_to_id = {n: i for i, n in cur.fetchall()}

    # Align Gator preds to train_activity compound order
    gator_lookup = dict(zip(df["compound_id"], df["Predicted_Affinity"]))
    y = y_df["pec50"].to_numpy(dtype=np.float64)
    gator_raw = np.asarray(
        [gator_lookup.get(int(cid), np.nan) for cid in y_df["compound_id"]],
        dtype=np.float64,
    )
    mask = np.isfinite(gator_raw)
    n_missing = int((~mask).sum())
    y_m = y[mask]
    g_m = gator_raw[mask]

    print(f"Train rows: {len(y)}  with-Gator: {mask.sum()}  missing: {n_missing}\n")

    # --- 1. Gator vs truth (zero-shot)
    pr_raw = stats.pearsonr(y_m, g_m)
    sp_raw = stats.spearmanr(y_m, g_m)
    print("=== Gator zero-shot vs pEC50 ===")
    print(f"Pearson  {pr_raw.statistic:+.4f}  (p={pr_raw.pvalue:.1e})")
    print(f"Spearman {sp_raw.statistic:+.4f}  (p={sp_raw.pvalue:.1e})")
    print(
        f"MAE raw      {mae(y_m, g_m):.4f}   (pred mean "
        f"{g_m.mean():.3f}, true mean {y_m.mean():.3f})"
    )

    # --- 2. Linear recalibration on full train (optimistic upper bound)
    reg = LinearRegression()
    reg.fit(g_m.reshape(-1, 1), y_m)
    pred_cal = reg.predict(g_m.reshape(-1, 1))
    print(
        f"MAE lin_cal  {mae(y_m, pred_cal):.4f}   (a={reg.coef_[0]:+.3f}, "
        f"b={reg.intercept_:+.3f})"
    )
    print(f"RAE lin_cal  {rae(y_m, pred_cal):.4f}\n")

    # --- 3. Pairwise Pearson vs existing OOFs
    print("=== Pairwise Pearson (full train, 4139 rows) ===")
    print(f"{'model':<44} {'r(Gator, model)':>17} {'r(model, y)':>13} {'OOF MAE':>8}")
    print("-" * 85)

    # Gator raw is fine for correlation (offset/scale don't affect r)
    for name in POOL:
        if name not in name_to_id:
            print(f"{name:<44}   (missing in DB)")
            continue
        oof = load_oof_predictions(name_to_id[name])
        # align mask
        oof_m = oof[mask]
        r_gm = stats.pearsonr(g_m, oof_m).statistic
        r_my = stats.pearsonr(oof_m, y_m).statistic
        oof_mae = mae(y_m, oof_m)
        print(f"{name:<44} {r_gm:>+17.4f} {r_my:>+13.4f} {oof_mae:>8.4f}")

    # --- 4. Linear-calibrated vs OOFs (fairer MAE comparison)
    print("\n=== MAE comparison (Gator linear-calibrated vs pool members) ===")
    print(f"  Gator_cal    MAE={mae(y_m, pred_cal):.4f}")
    for name in POOL:
        if name not in name_to_id:
            continue
        oof = load_oof_predictions(name_to_id[name])[mask]
        print(f"  {name:<42} MAE={mae(y_m, oof):.4f}")

    conn.close()


if __name__ == "__main__":
    main()
