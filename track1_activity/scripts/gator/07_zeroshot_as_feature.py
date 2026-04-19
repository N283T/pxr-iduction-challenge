"""Quick test: add Gator zero-shot affinity predictions as features
to 2d_full and train LightGBM to check whether Gator brings any
orthogonal signal to the 2D descriptor baseline.

Comparison:
  - baseline: 2d_full (mordred 1531 + jazzy 6 + rdkit_full 217 = 1754)
  - +gator:   baseline + [gator_pred_ft, gator_pred_pretrain]

Reports per-fold MAE/RAE, aggregate OOF metrics, and LightGBM gain
importance of the Gator columns (to see whether the tree splits on
them at all, and how high they rank).

Exploratory only -- does not write to experiments table. If the
+gator variant lowers OOF MAE meaningfully, bake the feature in via
a `2d_full_gator` bundle in run_train.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import (  # noqa: E402
    DB_PARAMS,
    JAZZY_FEATURE_COLS,
    load_jazzy,
    load_rdkit_full,
    load_train_mordred,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402


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


def load_boltz_tier0(train_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    """Pull the same Tier-0 scalar bundle as 2d_full_boltz."""
    conn = psycopg2.connect(**DB_PARAMS)
    q = "SELECT compound_id, " + ", ".join(TIER0_COLS) + " FROM compound_boltz2"
    df = pd.read_sql(q, conn).set_index("compound_id")
    conn.close()
    X = df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, [f"boltz:{c}" for c in TIER0_COLS]


GATOR_FT_CSV = REPO_ROOT.joinpath("structures", "gator", "preds_zero_shot_ft.csv")
GATOR_PRE_CSV = REPO_ROOT.joinpath(
    "structures", "gator", "preds_zero_shot_pretrain.csv"
)


def load_gator_csv(path: Path) -> dict[int, float]:
    df = pd.read_csv(path)
    cid = df["PDB_ID"].astype(str).str.split("_").str[0].astype(int)
    return dict(zip(cid, df["Predicted_Affinity"].astype(float)))


def build_2d_full(
    train_ids: list[int],
) -> tuple[np.ndarray, list[str]]:
    mordred_train, _ = load_train_mordred()
    Xm = mordred_train.loc[train_ids].to_numpy(dtype=np.float32)
    Xm = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0)
    m_cols = [f"mordred:{c}" for c in mordred_train.columns]

    jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
    Xj = jazzy_train[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
    j_cols = [f"jazzy:{c}" for c in JAZZY_FEATURE_COLS]

    rdkit_train = load_rdkit_full(train_ids)
    Xr = rdkit_train.loc[train_ids].to_numpy(dtype=np.float32)
    Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)
    r_cols = [f"rdkit:{c}" for c in rdkit_train.columns]

    X = np.concatenate([Xm, Xj, Xr], axis=1)
    cols = m_cols + j_cols + r_cols
    return X, cols


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rae(y, p):
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def _sanitize(names: list[str]) -> list[str]:
    # LGBM rejects JSON special chars in feature names.
    return [n.replace(":", "__").replace(" ", "_") for n in names]


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits,
    feature_names: list[str] | None = None,
    tag: str = "",
) -> tuple[np.ndarray, list[dict]]:
    if feature_names is not None:
        feature_names = _sanitize(feature_names)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_metrics = []
    importances = []
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
        train_ds = lgb.Dataset(X[tr], y[tr], feature_name=feature_names)
        val_ds = lgb.Dataset(
            X[va], y[va], reference=train_ds, feature_name=feature_names
        )
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=3000,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X[va])
        oof[va] = pred
        fm = {"fold": k, "mae": mae(y[va], pred), "rae": rae(y[va], pred)}
        fold_metrics.append(fm)
        if feature_names is not None:
            imp = model.feature_importance(importance_type="gain")
            importances.append(imp)
    print(f"\n=== {tag} ===")
    for fm in fold_metrics:
        print(f"  fold{fm['fold']}: MAE={fm['mae']:.4f}  RAE={fm['rae']:.4f}")
    print(
        f"  AVG      MAE={np.mean([f['mae'] for f in fold_metrics]):.4f}  "
        f"RAE={np.mean([f['rae'] for f in fold_metrics]):.4f}"
    )
    print(f"  OOF      MAE={mae(y, oof):.4f}  RAE={rae(y, oof):.4f}")
    if importances and feature_names is not None:
        avg_imp = np.mean(importances, axis=0)
        order = np.argsort(-avg_imp)
        print(f"\n  Top 10 features by mean gain:")
        for rk in order[:10]:
            print(f"    {feature_names[rk]:<40} {avg_imp[rk]:>10.1f}")
        # Spotlight gator columns if present
        gator_idx = [i for i, n in enumerate(feature_names) if n.startswith("gator:")]
        if gator_idx:
            print(f"\n  Gator columns:")
            for i in gator_idx:
                rank = int(np.where(order == i)[0])
                print(
                    f"    {feature_names[i]:<40} gain={avg_imp[i]:>10.1f}  "
                    f"rank={rank + 1}/{len(feature_names)}"
                )
    return oof, fold_metrics


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        "SELECT ta.compound_id, c.smiles, ta.pec50 "
        "FROM train_activity ta JOIN compounds c ON c.id = ta.compound_id "
        "ORDER BY ta.id",
        conn,
    )
    conn.close()
    train_ids = df["compound_id"].tolist()
    y = df["pec50"].to_numpy(dtype=np.float64)

    gator_ft = load_gator_csv(GATOR_FT_CSV)
    gator_pre = load_gator_csv(GATOR_PRE_CSV)

    # Align with train_ids; Auranofin (1657) is missing -> impute with
    # train mean so LightGBM doesn't reject the row. (Mean imputation
    # is safe because only one row is affected.)
    def _align(lookup: dict[int, float]) -> np.ndarray:
        vals = [lookup.get(int(c), np.nan) for c in train_ids]
        arr = np.asarray(vals, dtype=np.float64)
        mean = np.nanmean(arr)
        arr[~np.isfinite(arr)] = mean
        return arr

    x_ft = _align(gator_ft)
    x_pre = _align(gator_pre)
    print(f"train_ids={len(train_ids)}  y range [{y.min():.2f}, {y.max():.2f}]")
    print(f"gator_ft  [{x_ft.min():.2f}, {x_ft.max():.2f}] mean={x_ft.mean():.2f}")
    print(f"gator_pre [{x_pre.min():.2f}, {x_pre.max():.2f}] mean={x_pre.mean():.2f}")

    # 5-fold UMAP split, canonical seed=42
    smiles = df["smiles"].tolist()
    splits = umap_split_indices(smiles, n_splits=5, seed=42)

    print("\nLoading 2d_full feature matrix...")
    X_2d, col_2d = build_2d_full(train_ids)
    print(f"  shape {X_2d.shape}")

    X_plus = np.concatenate([X_2d, x_ft[:, None], x_pre[:, None]], axis=1)
    col_plus = col_2d + ["gator:pred_ft", "gator:pred_pretrain"]
    assert X_plus.shape[1] == len(col_plus)

    print(f"  baseline dim = {X_2d.shape[1]}")
    print(f"  +gator  dim  = {X_plus.shape[1]}")

    # Also build 2d_full + boltz tier-0 (subset of 2d_full_boltz) to test
    # whether gator brings anything beyond Boltz-2's own affinity head.
    print("\nLoading Boltz-2 Tier-0 scalars...")
    X_b, col_b = load_boltz_tier0(train_ids)
    print(f"  boltz tier-0 shape {X_b.shape}")

    X_2d_b = np.concatenate([X_2d, X_b], axis=1)
    col_2d_b = col_2d + col_b
    X_2d_b_plus = np.concatenate([X_2d_b, x_ft[:, None], x_pre[:, None]], axis=1)
    col_2d_b_plus = col_2d_b + ["gator:pred_ft", "gator:pred_pretrain"]

    # A/B/C/D CV
    _ = run_cv(X_2d, y, splits, feature_names=col_2d, tag="A: 2d_full")
    _ = run_cv(X_plus, y, splits, feature_names=col_plus, tag="B: 2d_full + gator")
    _ = run_cv(
        X_2d_b, y, splits, feature_names=col_2d_b, tag="C: 2d_full + boltz_tier0"
    )
    _ = run_cv(
        X_2d_b_plus,
        y,
        splits,
        feature_names=col_2d_b_plus,
        tag="D: 2d_full + boltz_tier0 + gator",
    )


if __name__ == "__main__":
    main()
