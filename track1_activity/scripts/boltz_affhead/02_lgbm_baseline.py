"""LightGBM baseline on pooled Boltz-2 trunk embeddings.

5-fold UMAP CV, canonical seed=42. Compares three input configurations:
  A: s_prot_mean + s_lig_mean + z_if_mean + z_if_max  (1024 dim, full)
  B: s_lig_mean + z_if_mean + z_if_max                 ( 640 dim, drop global protein mean)
  C: z_if_mean + z_if_max only                         ( 256 dim, interface-only)

Reference targets:
  - pool best tabpfn_2d_full_boltz OOF MAE = 0.4822 (includes tier-0)
  - LGBM C (2d_full + boltz_tier0) OOF MAE  = 0.5284 (LGBM + tier-0)

Decision gate:
  - A OOF MAE <= 0.50 -> strong, proceed to MLP head + ensemble
  - 0.50-0.55        -> weak but ensemble-able, test with pool correlation
  - > 0.55           -> head signal redundant with tier-0; stop
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402


POOLED_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled.parquet")


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rae(y, p):
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def run_cv(X, y, splits, tag: str) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    fold_mae = []
    params = dict(
        objective="regression",
        metric="mae",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=20,
        verbose=-1,
    )
    for k, (tr, va) in enumerate(splits):
        d_tr = lgb.Dataset(X[tr], y[tr])
        d_va = lgb.Dataset(X[va], y[va], reference=d_tr)
        model = lgb.train(
            params,
            d_tr,
            num_boost_round=3000,
            valid_sets=[d_va],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X[va])
        oof[va] = pred
        fold_mae.append(mae(y[va], pred))
    print(f"\n=== {tag} ===")
    for k, m in enumerate(fold_mae):
        print(f"  fold{k}: MAE={m:.4f}")
    print(f"  AVG  MAE={np.mean(fold_mae):.4f}")
    print(f"  OOF  MAE={mae(y, oof):.4f}  RAE={rae(y, oof):.4f}")
    pr = stats.pearsonr(y, oof)
    print(f"  Pearson={pr.statistic:+.4f}")
    return oof


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    ta = pd.read_sql(
        "SELECT compound_id, pec50 FROM train_activity ORDER BY id", conn
    )
    conn.close()

    pool = pd.read_parquet(POOLED_PATH)
    print(f"Pooled rows: {len(pool)}  cols: {len(pool.columns) - 1}")

    df = ta.merge(pool, on="compound_id", how="inner").sort_values(
        "compound_id"
    ).reset_index(drop=True)
    print(f"After inner join with train_activity: {len(df)} rows "
          f"(train was {len(ta)})")

    y = df["pec50"].to_numpy(dtype=np.float64)

    # Get SMILES for UMAP split
    conn = psycopg2.connect(**DB_PARAMS)
    smi_df = pd.read_sql(
        "SELECT id, std_smiles FROM compounds", conn
    ).set_index("id")
    conn.close()
    smiles = [smi_df.loc[int(c), "std_smiles"] for c in df["compound_id"]]
    splits = umap_split_indices(smiles, n_splits=5, seed=42)

    # Column groups
    col_s_prot = [c for c in df.columns if c.startswith("s_prot_mean_")]
    col_s_lig = [c for c in df.columns if c.startswith("s_lig_mean_")]
    col_z_mean = [c for c in df.columns if c.startswith("z_if_mean_")]
    col_z_max = [c for c in df.columns if c.startswith("z_if_max_")]
    print(
        f"  s_prot={len(col_s_prot)} s_lig={len(col_s_lig)} "
        f"z_mean={len(col_z_mean)} z_max={len(col_z_max)}"
    )

    cfgs = {
        "A: s_prot + s_lig + z_mean + z_max (1024)": col_s_prot + col_s_lig + col_z_mean + col_z_max,
        "B: s_lig + z_mean + z_max (640)": col_s_lig + col_z_mean + col_z_max,
        "C: z_mean + z_max (256)": col_z_mean + col_z_max,
    }
    for tag, cols in cfgs.items():
        X = df[cols].to_numpy(dtype=np.float32)
        _ = run_cv(X, y, splits, tag=tag)


if __name__ == "__main__":
    main()
