#!/usr/bin/env python
"""Bake-off: add Tier-2 IFP features on top of Tier-0+1 Boltz-2 + Mordred.

Runs 4 feature sets under the canonical UMAP split with default LGBM:
  1. mordred
  2. mordred + boltz2_tier0 (scalars)
  3. mordred + boltz2_tier0 + tier1 (confidence maps)
  4. mordred + boltz2_tier0 + tier1 + tier2 (ProLIF IFP)
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.joinpath("src")))

from data import get_conn, load_train_mordred, load_train_smiles_target  # noqa: E402
from splits import umap_split_indices  # noqa: E402

DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")
REPORT_DIR = ROOT.joinpath("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TIER1_PARQUET = ROOT.parent.joinpath("data", "boltz2_confidence_features.parquet")
TIER2_PARQUET = ROOT.parent.joinpath("data", "boltz2_ifp_features.parquet")

BOLTZ2_SCALAR_COLS = [
    "affinity_pred_value",
    "affinity_pred_value_1",
    "affinity_pred_value_2",
    "affinity_probability_binary",
    "affinity_probability_binary_1",
    "affinity_probability_binary_2",
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "ligand_atom_count",
    "ligand_to_pocket_distance_a",
]


def load_train_compound_ids() -> list[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def load_boltz2_tier0(ids: list[int]) -> tuple[np.ndarray, list[str]]:
    cols_sql = ", ".join(f"b.{c}" for c in BOLTZ2_SCALAR_COLS)
    query = f"""
    SELECT c.id AS compound_id, {cols_sql},
           (b.affinity_pred_value_1 - b.affinity_pred_value_2) AS ensemble_diff_affinity,
           (b.affinity_probability_binary_1 - b.affinity_probability_binary_2) AS ensemble_diff_prob
    FROM compounds c
    LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
    WHERE c.id = ANY(%s) ORDER BY c.id
    """
    with psycopg2.connect(**DB_KW) as conn:
        df = pd.read_sql(query, conn, params=(ids,))
    df = df.set_index("compound_id").reindex(ids)
    names = list(BOLTZ2_SCALAR_COLS) + ["ensemble_diff_affinity", "ensemble_diff_prob"]
    return df[names].to_numpy(dtype=np.float32), names


def load_tier(parquet_path: Path, ids: list[int]) -> tuple[np.ndarray, list[str]]:
    df = pd.read_parquet(parquet_path).reindex(ids)
    names = list(df.columns)
    return df.to_numpy(dtype=np.float32), names


def lgbm_cv_oof(X, y, splits, feature_names, params):
    oof = np.full(len(y), np.nan, dtype=np.float32)
    boosters = []
    for fold_idx, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=feature_names)
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr, feature_name=feature_names)
        bo = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = bo.predict(X[va], num_iteration=bo.best_iteration)
        boosters.append(bo)
        print(
            f"  fold {fold_idx}: n_val={len(va)}, MAE={float(np.mean(np.abs(y[va] - oof[va]))):.4f}"
        )
    return oof, boosters


def metrics(y, p):
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RAE": float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - np.mean(y)))),
        "Pearson": float(pearsonr(y, p)[0]),
    }


def main():
    train_df = load_train_smiles_target()
    train_ids = load_train_compound_ids()
    smiles = train_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"Train: {len(train_ids)}")

    mordred_train, _ = load_train_mordred()
    mord_cols = list(mordred_train.columns)
    X_mord = np.nan_to_num(
        mordred_train.loc[train_ids, mord_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_t0, n_t0 = load_boltz2_tier0(train_ids)
    X_t1, n_t1 = load_tier(TIER1_PARQUET, train_ids)
    X_t2, n_t2 = load_tier(TIER2_PARQUET, train_ids)
    # Fill NaN in IFP (missing compounds) with 0 (no interaction)
    X_t2 = np.nan_to_num(X_t2, nan=0.0)
    print(f"Tier-0 scalars: {X_t0.shape}")
    print(f"Tier-1 conf:    {X_t1.shape}")
    print(f"Tier-2 IFP:     {X_t2.shape}")

    X_mt0 = np.concatenate([X_mord, X_t0], axis=1)
    n_mt0 = mord_cols + n_t0
    X_mt01 = np.concatenate([X_mord, X_t0, X_t1], axis=1)
    n_mt01 = mord_cols + n_t0 + n_t1
    X_mt012 = np.concatenate([X_mord, X_t0, X_t1, X_t2], axis=1)
    n_mt012 = mord_cols + n_t0 + n_t1 + n_t2

    print("\n=== UMAP split (canonical) ===")
    splits = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)

    params = dict(
        objective="regression",
        metric="mae",
        boosting_type="gbdt",
        verbose=-1,
        seed=42,
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=20,
    )

    # Boltz-only variants (for orthogonality check vs Mordred)
    X_t01 = np.concatenate([X_t0, X_t1], axis=1)
    n_t01 = n_t0 + n_t1
    X_t012 = np.concatenate([X_t0, X_t1, X_t2], axis=1)
    n_t012 = n_t0 + n_t1 + n_t2

    runs = [
        ("mordred", X_mord, mord_cols),
        ("boltz_tier0", X_t0, n_t0),
        ("boltz_tier0+tier1", X_t01, n_t01),
        ("boltz_tier0+tier1+tier2", X_t012, n_t012),
        ("mordred+tier0", X_mt0, n_mt0),
        ("mordred+tier0+tier1", X_mt01, n_mt01),
        ("mordred+tier0+tier1+tier2", X_mt012, n_mt012),
    ]
    results = {}
    importances = {}
    for name, X, fn in runs:
        print(f"\n=== {name}  ({X.shape[1]} features) ===")
        oof, boosters = lgbm_cv_oof(X, y, splits, fn, params)
        m = metrics(y, oof)
        results[name] = m
        print(
            f"  OOF MAE={m['MAE']:.4f}  RAE={m['RAE']:.4f}  Pearson={m['Pearson']:.4f}"
        )
        imp = np.zeros(X.shape[1], dtype=np.float64)
        for bo in boosters:
            imp += bo.feature_importance(importance_type="gain")
        importances[name] = pd.DataFrame({"feature": fn, "gain_sum": imp}).sort_values(
            "gain_sum", ascending=False
        )

    print("\n=== Summary ===")
    summary = pd.DataFrame(results).T.round(4)
    print(summary.to_string())
    summary.to_csv(REPORT_DIR.joinpath("boltz2_ifp_bakeoff_summary.csv"))

    # Top 25 + tier-2 ranks for the final model
    plus = importances["mordred+tier0+tier1+tier2"].reset_index(drop=True)
    plus["rank"] = plus.index + 1
    print("\n=== Top 25 features (final model) ===")
    print(plus.head(25).to_string(index=False))
    tier2_rank = plus[plus["feature"].isin(n_t2)].sort_values("rank")
    nonzero = tier2_rank[tier2_rank["gain_sum"] > 0]
    print(
        f"\n=== Tier-2 IFP feature ranks (total {len(plus)}, "
        f"{len(nonzero)} IFP with gain>0) ==="
    )
    print(nonzero.head(30).to_string(index=False))

    plus.to_csv(
        REPORT_DIR.joinpath("boltz2_ifp_bakeoff_importance_final.csv"), index=False
    )
    tier2_rank.to_csv(
        REPORT_DIR.joinpath("boltz2_ifp_bakeoff_tier2_ranks.csv"), index=False
    )
    print("\nSaved to reports/")


if __name__ == "__main__":
    main()
