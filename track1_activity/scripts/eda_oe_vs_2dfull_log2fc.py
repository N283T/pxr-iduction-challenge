"""4-variant LGBM ablation: 2d_full_boltz_log2fc_pred ± xlogp ± ROCS.

Base = production tabpfn_2d_full_boltz_log2fc_pred feature space (1803 dim).
Tests whether (a) xlogp alone, (b) ROCS alone, (c) both add value. The
LogP-only ablation lets us judge whether xlogp is a shortcut vs a genuine
addition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import lightgbm as lgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import run_train  # noqa: E402


ROCS_COLS = [
    "max_shape_tanimoto",
    "max_color_tanimoto",
    "max_combo_tanimoto",
    "mean_shape_tanimoto",
    "mean_color_tanimoto",
    "mean_combo_tanimoto",
]


def load_train_test_dfs():
    conn = psycopg2.connect(**DB_PARAMS)
    tr = pd.read_sql(
        """SELECT t.compound_id AS id, c.std_smiles AS smiles, t.pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        conn,
    )
    te = pd.read_sql(
        """SELECT t.compound_id AS id, c.std_smiles AS smiles
           FROM test_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        conn,
    )
    conn.close()
    return (
        tr,
        te,
        tr["pec50"].to_numpy(dtype=float),
        tr["smiles"].tolist(),
    )


def load_oe_slim(compound_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    conn = psycopg2.connect(**DB_PARAMS)
    oe = pd.read_sql(
        "SELECT compound_id, xlogp FROM compound_oemedchem WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    rocs = pd.read_sql(
        f"SELECT compound_id, {','.join(ROCS_COLS)} "
        f"FROM compound_rocs WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    conn.close()
    oe = oe.reindex(compound_ids)
    rocs = rocs.reindex(compound_ids)

    X_xlogp = np.array(oe[["xlogp"]].to_numpy(dtype=np.float32), copy=True)
    X_rocs = np.array(rocs[ROCS_COLS].to_numpy(dtype=np.float32), copy=True)
    # Mean-fill (rare NaN in ROCS where omega failed)
    for X in (X_xlogp, X_rocs):
        col_means = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X_xlogp, X_rocs


def cv(X: np.ndarray, y: np.ndarray, fold_idx: np.ndarray, tag: str) -> dict:
    oof = np.zeros_like(y)
    total_gain = np.zeros(X.shape[1])
    for fold in range(5):
        tr = fold_idx != fold
        va = fold_idx == fold
        m = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
        )
        m.fit(
            X[tr],
            y[tr],
            eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof[va] = m.predict(X[va])
        total_gain += m.booster_.feature_importance(importance_type="gain")
    mae = float(np.mean(np.abs(oof - y)))
    rae = mae / float(np.mean(np.abs(y - y.mean())))
    sp = float(spearmanr(oof, y).statistic)
    return {
        "tag": tag,
        "n": X.shape[1],
        "mae": mae,
        "rae": rae,
        "sp": sp,
        "total_gain": total_gain,
    }


def main() -> None:
    print("Loading train/test splits + 2d_full_boltz_log2fc_pred...")
    train_df, test_df, y, smiles_list = load_train_test_dfs()
    train_ids = train_df["id"].tolist()

    X_base_tr, _ = run_train.load_features(
        "2d_full_boltz_log2fc_pred", train_df, test_df
    )
    print(f"  base 2d_full_boltz_log2fc_pred: {X_base_tr.shape}")

    X_xlogp, X_rocs = load_oe_slim(train_ids)
    print(f"  xlogp slim: {X_xlogp.shape}  rocs slim: {X_rocs.shape}")

    print("\nComputing UMAP 5-fold split (Morgan+Jaccard, k=50, seed=42)...")
    splits = umap_split_indices(smiles_list, n_splits=5, n_clusters=50, seed=42)
    fold_idx = np.zeros(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    variants = {
        "base  (2d_full_boltz_log2fc_pred)": X_base_tr,
        "+xlogp only": np.concatenate([X_base_tr, X_xlogp], axis=1),
        "+rocs only (no LogP)": np.concatenate([X_base_tr, X_rocs], axis=1),
        "+xlogp+rocs": np.concatenate([X_base_tr, X_xlogp, X_rocs], axis=1),
    }

    print("\n5-fold OOF (LGBM default):")
    print(f"  {'variant':<40s} {'n':<6s} {'MAE':<8s} {'RAE':<8s} {'Sp':<8s}")
    print("  " + "-" * 75)
    results = {}
    for tag, X in variants.items():
        r = cv(X, y, fold_idx, tag)
        results[tag] = r
        print(
            f"  {r['tag']:<40s} {r['n']:<6d} "
            f"{r['mae']:.4f}  {r['rae']:.4f}  {r['sp']:.4f}"
        )

    # Gain share of added features in each ablation
    print("\nGain share of ADDED OE features (only) per variant:")
    base_n = X_base_tr.shape[1]
    for tag, r in results.items():
        if r["n"] == base_n:
            continue
        added_gain = r["total_gain"][base_n:]
        total = r["total_gain"].sum()
        share = added_gain.sum() / total * 100
        print(
            f"  {r['tag']:<40s}  added n={r['n'] - base_n}  "
            f"share={share:.2f}%  added gain={added_gain.sum():.0f}"
        )

    # Delta MAE vs base
    base_mae = results["base  (2d_full_boltz_log2fc_pred)"]["mae"]
    print("\nΔ MAE vs base:")
    for tag, r in results.items():
        if r["tag"] == "base  (2d_full_boltz_log2fc_pred)":
            continue
        d = r["mae"] - base_mae
        print(f"  {r['tag']:<40s}  ΔMAE = {d:+.4f}")


if __name__ == "__main__":
    main()
