"""Combine pooled Boltz-2 trunk embeddings with 2d_full / boltz_tier0
features and compare against the 8-model ensemble pool for diversity.

Configurations (5-fold UMAP CV, canonical seed=42, LightGBM):
  X1: 2d_full (1754)                   -- reference baseline
  X2: 2d_full + pooled_boltz (2778)    -- trunk embeddings on top
  X3: 2d_full + boltz_tier0 (1771)     -- current LGBM best with Boltz scalars
  X4: 2d_full + tier0 + pooled (2795)  -- trunk + scalars
  X5: pooled_boltz alone (1024)        -- for correlation study

For each non-reference config we also save the OOF predictions and
report Pearson correlation against each 8-model pool member.
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
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    JAZZY_FEATURE_COLS,
    load_jazzy,
    load_rdkit_full,
    load_train_mordred,
)
from evaluate import load_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402


POOLED_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled.parquet")
OOF_OUT = REPO_ROOT.joinpath("data", "boltz_affhead", "oof_lgbm_pooled.parquet")


TIER0_COLS = (
    "ligand_atom_count",
    "ligand_to_pocket_distance_a",
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
)


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


def run_cv(X, y, splits, tag: str) -> np.ndarray:
    oof = np.full(len(y), np.nan)
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
    fold_mae = []
    for k, (tr, va) in enumerate(splits):
        d_tr = lgb.Dataset(X[tr], y[tr])
        d_va = lgb.Dataset(X[va], y[va], reference=d_tr)
        model = lgb.train(
            params,
            d_tr,
            num_boost_round=3000,
            valid_sets=[d_va],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred = model.predict(X[va])
        oof[va] = pred
        fold_mae.append(mae(y[va], pred))
    pr = stats.pearsonr(y, oof)
    print(
        f"  [{tag}] dim={X.shape[1]:<5}  "
        f"AVG_MAE={np.mean(fold_mae):.4f}  "
        f"OOF_MAE={mae(y, oof):.4f}  "
        f"OOF_RAE={rae(y, oof):.4f}  "
        f"Pearson={pr.statistic:+.4f}"
    )
    return oof


def load_boltz_tier0(train_ids):
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        "SELECT compound_id, " + ", ".join(TIER0_COLS) + " FROM compound_boltz2",
        conn,
    ).set_index("compound_id")
    conn.close()
    X = df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def load_2d_full(train_ids):
    mordred_train, _ = load_train_mordred()
    Xm = mordred_train.loc[train_ids].to_numpy(dtype=np.float32)
    Xm = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0)

    jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
    Xj = jazzy_train[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)

    rdkit_train = load_rdkit_full(train_ids)
    Xr = rdkit_train.loc[train_ids].to_numpy(dtype=np.float32)
    Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate([Xm, Xj, Xr], axis=1)


def load_pooled(train_ids):
    pool = pd.read_parquet(POOLED_PATH).set_index("compound_id")
    X = pool.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
    # Auranofin (cid 1657) has no embedding; impute with column mean.
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def pool_correlation(oof_pooled, y):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (list(POOL),),
    )
    name_to_id = {n: i for i, n in cur.fetchall()}
    conn.close()

    print(
        f"\n{'model':<42} {'r(pooled, model)':>18} {'r(model, y)':>13} {'OOF MAE':>9}"
    )
    print("-" * 85)
    for name in POOL:
        if name not in name_to_id:
            continue
        oof_other = load_oof_predictions(name_to_id[name])
        r_gm = stats.pearsonr(oof_pooled, oof_other).statistic
        r_my = stats.pearsonr(oof_other, y).statistic
        print(f"{name:<42} {r_gm:>+18.4f} {r_my:>+13.4f} {mae(y, oof_other):>9.4f}")
    print("\n  Reference: r(pooled_lgbm, y) = Pearson reported above")


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    ta = pd.read_sql(
        "SELECT ta.compound_id, c.std_smiles, ta.pec50 "
        "FROM train_activity ta JOIN compounds c ON c.id = ta.compound_id "
        "ORDER BY ta.id",
        conn,
    )
    conn.close()
    train_ids = ta["compound_id"].tolist()
    y = ta["pec50"].to_numpy(dtype=np.float64)
    smiles = ta["std_smiles"].tolist()
    splits = umap_split_indices(smiles, n_splits=5, seed=42)

    print("Loading features...")
    X_2d = load_2d_full(train_ids)
    X_b0 = load_boltz_tier0(train_ids)
    X_po = load_pooled(train_ids)
    print(f"  2d_full={X_2d.shape} tier0={X_b0.shape} pooled={X_po.shape}")

    print("\n=== A/B/C/D/E LightGBM comparison ===")
    oof_2d = run_cv(X_2d, y, splits, "X1 2d_full")
    oof_2d_po = run_cv(
        np.concatenate([X_2d, X_po], axis=1), y, splits, "X2 2d_full + pooled"
    )
    oof_2d_b0 = run_cv(
        np.concatenate([X_2d, X_b0], axis=1), y, splits, "X3 2d_full + tier0"
    )
    oof_2d_b0_po = run_cv(
        np.concatenate([X_2d, X_b0, X_po], axis=1),
        y,
        splits,
        "X4 2d_full + tier0 + pooled",
    )
    oof_po = run_cv(X_po, y, splits, "X5 pooled only")

    # Save the pooled-only OOF (X5) for downstream
    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"compound_id": train_ids, "oof_pooled": oof_po, "pec50": y})
    out_df.to_parquet(OOF_OUT, index=False)
    print(f"\nSaved OOF (X5 pooled) -> {OOF_OUT}")

    print("\n=== Pool correlation of X5 (pooled only) ===")
    pool_correlation(oof_po, y)


if __name__ == "__main__":
    main()
