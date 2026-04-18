#!/usr/bin/env python
"""LGBM OOF bake-off: Mordred vs Boltz-2 vs Mordred+Boltz-2.

Goal: quick signal-check. Does adding Boltz-2 scalar features to a Mordred
LightGBM baseline improve OOF MAE / RAE under the canonical UMAP split?
Also dumps top feature importance for the combined model.

Usage:
    pixi run python track1_activity/scripts/eda_boltz2_feature_bakeoff.py
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

from data import (  # noqa: E402
    get_conn,
    load_rdkit_full,
    load_train_mordred,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402


def load_train_compound_ids() -> list[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")
REPORT_DIR = ROOT.joinpath("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BOLTZ2_NUMERIC_COLS = [
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

# PoseBusters booleans dropped — all gain=0 in first bake-off (PXR compounds
# are uniformly drug-like, so every check passes on essentially every row).
# Kept only num_passed as a single summary (may still offer weak signal).
POSEBUSTERS_COLS = ["num_passed"]


def load_boltz2_features(compound_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    """Pull Boltz-2 scalar features + PoseBusters booleans for given IDs.

    Returns (X matrix [n, d], feature_names) with NaN preserved for missing
    rows (LightGBM handles natively).
    """
    numeric_sql = ", ".join(f"b.{c}" for c in BOLTZ2_NUMERIC_COLS)
    pose_sql = ", ".join(f"pb.{c}" for c in POSEBUSTERS_COLS)
    derived_sql = """
    , (b.affinity_pred_value_1 - b.affinity_pred_value_2) AS ensemble_diff_affinity
    , (b.affinity_probability_binary_1 - b.affinity_probability_binary_2)
        AS ensemble_diff_prob
    """
    query = f"""
    SELECT c.id AS compound_id,
           {numeric_sql},
           {pose_sql}
           {derived_sql}
    FROM compounds c
    LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
    LEFT JOIN compound_boltz2_posebusters pb ON pb.compound_id = c.id
    WHERE c.id = ANY(%s)
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_KW) as conn:
        df = pd.read_sql(query, conn, params=(compound_ids,))
    df = df.set_index("compound_id").reindex(compound_ids)

    # Cast booleans to float (NaN preserved for missing)
    for col in POSEBUSTERS_COLS:
        if df[col].dtype == object:
            df[col] = df[col].map({True: 1.0, False: 0.0})
        df[col] = df[col].astype("float32")

    feature_names = (
        list(BOLTZ2_NUMERIC_COLS)
        + list(POSEBUSTERS_COLS)
        + ["ensemble_diff_affinity", "ensemble_diff_prob"]
    )
    X = df[feature_names].to_numpy(dtype=np.float32)
    return X, feature_names


def lgbm_cv_oof(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    params: dict,
    seed: int = 42,
) -> tuple[np.ndarray, list[lgb.Booster]]:
    oof = np.full(len(y), np.nan, dtype=np.float32)
    boosters: list[lgb.Booster] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        dvalid = lgb.Dataset(
            X_va, label=y_va, reference=dtrain, feature_name=feature_names
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = booster.predict(X_va, num_iteration=booster.best_iteration)
        boosters.append(booster)
        fold_mae = float(np.mean(np.abs(y_va - oof[val_idx])))
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
    assert len(train_ids) == len(train_df), (
        f"ID/SMILES length mismatch: {len(train_ids)} vs {len(train_df)}"
    )
    smiles = train_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    print(
        f"Train compounds: {len(train_ids)}, pec50 range: {y.min():.2f}-{y.max():.2f}"
    )

    # Mordred
    mordred_train, _ = load_train_mordred()
    missing = set(train_ids) - set(mordred_train.index)
    if missing:
        raise ValueError(f"Mordred missing for {len(missing)} train ids")
    mord_cols = list(mordred_train.columns)
    X_mord = mordred_train.loc[train_ids, mord_cols].to_numpy(dtype=np.float32)
    X_mord = np.nan_to_num(X_mord, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Mordred: {X_mord.shape}")

    # rdkit_desc_full (217d)
    rdkit_full = load_rdkit_full(train_ids)
    missing = set(train_ids) - set(rdkit_full.index)
    if missing:
        raise ValueError(f"rdkit_desc_full missing for {len(missing)} train ids")
    rdkit_cols = list(rdkit_full.columns)
    X_rdkit = rdkit_full.loc[train_ids, rdkit_cols].to_numpy(dtype=np.float32)
    print(f"rdkit_desc_full: {X_rdkit.shape}  (NaN preserved for LightGBM)")

    # Boltz-2
    X_boltz, boltz_names = load_boltz2_features(train_ids)
    print(
        f"Boltz-2: {X_boltz.shape}, {np.isnan(X_boltz).any(axis=1).sum()} rows have "
        f"any NaN (kept as-is)"
    )

    # Combined matrices
    X_mb = np.concatenate([X_mord, X_boltz], axis=1)
    mb_names = list(mord_cols) + boltz_names
    print(f"Mordred+Boltz-2: {X_mb.shape}")

    X_rb = np.concatenate([X_rdkit, X_boltz], axis=1)
    rb_names = list(rdkit_cols) + boltz_names
    print(f"rdkit_desc_full+Boltz-2: {X_rb.shape}")

    # UMAP split (Morgan FP default)
    print("\n=== Building UMAP split (Morgan FP, 50 clusters, 5 folds) ===")
    splits = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)

    # LGBM params — canonical defaults from run_train.py
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
        ("rdkit_desc_full", X_rdkit, rdkit_cols),
        ("boltz2", X_boltz, boltz_names),
        ("mordred_boltz2", X_mb, mb_names),
        ("rdkit_desc_full_boltz2", X_rb, rb_names),
    ]:
        print(f"\n=== {name}  ({X.shape[1]} features) ===")
        oof, boosters = lgbm_cv_oof(X, y, splits, fn, params)
        m = metrics(y, oof)
        results[name] = m
        print(
            f"  OOF MAE={m['MAE']:.4f}  RAE={m['RAE']:.4f}  Pearson={m['Pearson']:.4f}"
        )

        # Aggregate gain importance across folds
        imp = np.zeros(X.shape[1], dtype=np.float64)
        for bo in boosters:
            imp += bo.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame(
            {"feature": fn, "gain_sum": imp, "gain_mean": imp / len(boosters)}
        ).sort_values("gain_sum", ascending=False)
        importances[name] = imp_df

    print("\n=== Summary (lower MAE / RAE is better) ===")
    summary = pd.DataFrame(results).T.round(4)
    print(summary.to_string())
    summary.to_csv(REPORT_DIR.joinpath("boltz2_bakeoff_summary.csv"))

    # Top importance + Boltz-2 feature ranks, for each combined model
    for combo in ("mordred_boltz2", "rdkit_desc_full_boltz2"):
        print(f"\n=== Top 20 features ({combo}, by gain) ===")
        top20 = importances[combo].head(20)
        print(top20.to_string(index=False))
        top20.to_csv(
            REPORT_DIR.joinpath(f"boltz2_bakeoff_top20_{combo}.csv"), index=False
        )

        imp = importances[combo].reset_index(drop=True)
        imp["rank"] = imp.index + 1
        boltz_rank = imp[imp["feature"].isin(boltz_names)].sort_values("rank")
        total = len(imp)
        print(f"\n=== Rank of Boltz-2 features within {combo} (total {total}) ===")
        print(boltz_rank.to_string(index=False))
        boltz_rank.to_csv(
            REPORT_DIR.joinpath(f"boltz2_bakeoff_boltz2_ranks_{combo}.csv"),
            index=False,
        )

    # Full importance tables
    for name, imp_df in importances.items():
        imp_df.to_csv(
            REPORT_DIR.joinpath(f"boltz2_bakeoff_importance_{name}.csv"),
            index=False,
        )

    print("\nAll outputs saved to reports/")


if __name__ == "__main__":
    main()
