#!/usr/bin/env python
"""LGBM OOF bake-off: add Tier-1 confidence features to the Mordred+Boltz-2 base.

Base: mordred + boltz2_scalars (19 scalar features, PoseBusters dropped).
Delta: + 44 Tier-1 confidence-map features (plddt/pae/pde aggregated over
protein / ligand / pocket / cross-blocks + per-residue pocket pLDDT).

Compares 3 feature sets under canonical UMAP split:
  - mordred
  - mordred + boltz2_scalars
  - mordred + boltz2_scalars + boltz2_conf (Tier-1)
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
CONF_PARQUET = ROOT.parent.joinpath("data", "boltz2_confidence_features.parquet")

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


def load_boltz2_scalars(compound_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    cols_sql = ", ".join(f"b.{c}" for c in BOLTZ2_SCALAR_COLS)
    derived_sql = """
    , (b.affinity_pred_value_1 - b.affinity_pred_value_2) AS ensemble_diff_affinity
    , (b.affinity_probability_binary_1 - b.affinity_probability_binary_2)
        AS ensemble_diff_prob
    """
    query = f"""
    SELECT c.id AS compound_id, {cols_sql} {derived_sql}
    FROM compounds c
    LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
    WHERE c.id = ANY(%s)
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_KW) as conn:
        df = pd.read_sql(query, conn, params=(compound_ids,))
    df = df.set_index("compound_id").reindex(compound_ids)
    feature_names = list(BOLTZ2_SCALAR_COLS) + [
        "ensemble_diff_affinity",
        "ensemble_diff_prob",
    ]
    return df[feature_names].to_numpy(dtype=np.float32), feature_names


def load_boltz2_conf(compound_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    df = pd.read_parquet(CONF_PARQUET).reindex(compound_ids)
    feature_names = list(df.columns)
    return df.to_numpy(dtype=np.float32), feature_names


def lgbm_cv_oof(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    params: dict,
) -> tuple[np.ndarray, list[lgb.Booster]]:
    oof = np.full(len(y), np.nan, dtype=np.float32)
    boosters: list[lgb.Booster] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        dtrain = lgb.Dataset(
            X[train_idx], label=y[train_idx], feature_name=feature_names
        )
        dvalid = lgb.Dataset(
            X[val_idx],
            label=y[val_idx],
            reference=dtrain,
            feature_name=feature_names,
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = booster.predict(X[val_idx], num_iteration=booster.best_iteration)
        boosters.append(booster)
        fold_mae = float(np.mean(np.abs(y[val_idx] - oof[val_idx])))
        print(f"  fold {fold_idx}: n_val={len(val_idx)}, MAE={fold_mae:.4f}")
    return oof, boosters


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y - p)))
    rae = float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - np.mean(y))))
    r, _ = pearsonr(y, p)
    return {"MAE": mae, "RAE": rae, "Pearson": float(r)}


def main() -> None:
    print("=== Loading data ===")
    train_df = load_train_smiles_target()
    train_ids = load_train_compound_ids()
    assert len(train_ids) == len(train_df)
    smiles = train_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"Train compounds: {len(train_ids)}")

    mordred_train, _ = load_train_mordred()
    mord_cols = list(mordred_train.columns)
    X_mord = np.nan_to_num(
        mordred_train.loc[train_ids, mord_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    print(f"Mordred: {X_mord.shape}")

    X_scalar, scalar_names = load_boltz2_scalars(train_ids)
    print(f"Boltz-2 scalars: {X_scalar.shape}")

    X_conf, conf_names = load_boltz2_conf(train_ids)
    print(
        f"Boltz-2 confidence (Tier-1): {X_conf.shape}, "
        f"{np.isnan(X_conf).any(axis=1).sum()} rows have any NaN"
    )

    X_base = np.concatenate([X_mord, X_scalar], axis=1)
    base_names = mord_cols + scalar_names

    X_plus = np.concatenate([X_mord, X_scalar, X_conf], axis=1)
    plus_names = mord_cols + scalar_names + conf_names

    print("\n=== Building UMAP split ===")
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

    results = {}
    importances = {}

    for name, X, fn in [
        ("mordred", X_mord, mord_cols),
        ("mordred_boltz2_scalars", X_base, base_names),
        ("mordred_boltz2_scalars_conf", X_plus, plus_names),
    ]:
        print(f"\n=== {name}  ({X.shape[1]} features) ===")
        oof, boosters = lgbm_cv_oof(X, y, splits, fn, params)
        m = metrics(y, oof)
        results[name] = m
        print(f"  OOF MAE={m['MAE']:.4f}  RAE={m['RAE']:.4f}  Pearson={m['Pearson']:.4f}")
        imp = np.zeros(X.shape[1], dtype=np.float64)
        for bo in boosters:
            imp += bo.feature_importance(importance_type="gain")
        importances[name] = pd.DataFrame(
            {"feature": fn, "gain_sum": imp}
        ).sort_values("gain_sum", ascending=False)

    print("\n=== Summary ===")
    summary = pd.DataFrame(results).T.round(4)
    print(summary.to_string())
    summary.to_csv(REPORT_DIR.joinpath("boltz2_conf_bakeoff_summary.csv"))

    # Top 25 and rank of confidence features in the 'plus' model
    plus_imp = importances["mordred_boltz2_scalars_conf"].reset_index(drop=True)
    plus_imp["rank"] = plus_imp.index + 1
    print("\n=== Top 25 features (plus model) ===")
    print(plus_imp.head(25).to_string(index=False))

    conf_rank = plus_imp[plus_imp["feature"].isin(conf_names)].sort_values("rank")
    print(
        f"\n=== Tier-1 confidence-feature ranks (total {len(plus_imp)} features) ==="
    )
    print(conf_rank.head(30).to_string(index=False))
    print(f"\n...{len(conf_rank)} total confidence features; bottom entries omitted")

    plus_imp.to_csv(
        REPORT_DIR.joinpath("boltz2_conf_bakeoff_importance_plus.csv"), index=False
    )
    conf_rank.to_csv(
        REPORT_DIR.joinpath("boltz2_conf_bakeoff_conf_ranks.csv"), index=False
    )
    print("\nSaved to reports/")


if __name__ == "__main__":
    main()
