#!/usr/bin/env -S pixi run python
"""All-inclusive LGBM feature importance for multitask aux target selection.

Combines every descriptor family we already compute per compound:
  - Mordred 2D (~1531)
  - Jazzy pose (6, compound_boltz2_jazzy)
  - Jazzy self (6, compound_jazzy -- keeps self vs pose comparable)
  - RDKit full (217, compound_descriptors_full)
  - Boltz-2 tier-0 scalars (19: 17 cols + 2 derived ensemble diffs)
  - Boltz-2 tier-1 confidence aggregates (44, from parquet)
  - 3D ligand (1212: scalar3d + 6 vector families + electroshape + mordred3d)

Trains default-hyperparam LGBM across the 5-fold UMAP split (seed 42).
Dumps gain-based feature importance averaged across folds + a stability
count. No Optuna, no ensemble: the goal is ranking for downstream
chemprop multitask aux-target selection, not SOTA single-model RAE.

Outputs:
  reports/multitask_aux/importance_all_desc.csv  -- per-feature gain/rank
  reports/multitask_aux/importance_all_desc.log  -- CV metrics + top-50

Run:
  pixi run python track1_activity/scripts/multitask_aux/01_importance_lgbm.py
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
    get_conn,
    load_jazzy,
    load_mordred,
    load_rdkit_full,
    load_test_smiles,
    load_train_mordred,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402


def load_compound_ids(split: str) -> list[int]:
    with get_conn() as conn:
        cur = conn.cursor()
        table = "train_activity" if split == "train" else "test_activity"
        cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
        return [r[0] for r in cur.fetchall()]


REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports", "multitask_aux")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LGBM_PARAMS = dict(
    objective="regression",
    metric="mae",
    boosting_type="gbdt",
    verbose=-1,
    seed=42,
    num_leaves=63,
    learning_rate=0.02,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    lambda_l1=0.01,
    lambda_l2=1.0,
    num_threads=0,
)

BOLTZ_TIER0_COLS = (
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
)

D3_SCALAR_COLS = (
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pmi1",
    "pmi2",
    "pmi3",
    "radius_of_gyration",
    "spherocity_index",
    "pbf",
)
D3_VECTOR_SPEC = (
    ("autocorr3d", 80),
    ("getaway", 273),
    ("morse", 224),
    ("rdf", 210),
    ("whim", 114),
    ("usr", 12),
    ("usrcat", 60),
)


def _sanitize(names: list[str]) -> list[str]:
    return [n.replace(":", "__").replace(",", "_") for n in names]


def _fill_vec(series: pd.Series, dim: int) -> np.ndarray:
    out = []
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.zeros(dim, dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float64)
            arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)
            out.append(arr.astype(np.float32))
    return np.stack(out, axis=0)


def _mord_matrix(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    cols = None
    for v in series:
        if v is not None:
            cols = sorted(v.keys())
            break
    if cols is None:
        raise RuntimeError("compound_boltz2_mordred3d is empty")
    rows = []
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            rows.append(np.zeros(len(cols), dtype=np.float32))
        else:
            vals = [v.get(c) for c in cols]
            arr = np.asarray(
                [float(x) if x is not None else 0.0 for x in vals],
                dtype=np.float64,
            )
            arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)
            rows.append(arr.astype(np.float32))
    return np.stack(rows, axis=0), cols


def build_matrix(train_ids: list[int], test_ids: list[int]):
    blocks_tr: list[np.ndarray] = []
    blocks_te: list[np.ndarray] = []
    names: list[str] = []

    # Mordred 2D (~1531)
    mordred_train, _ = load_train_mordred()
    mordred_test = load_mordred(test_ids)
    common_m = sorted(mordred_train.columns.intersection(mordred_test.columns))
    Xm_tr = mordred_train.loc[train_ids, common_m].to_numpy(dtype=np.float32)
    Xm_te = mordred_test.loc[test_ids, common_m].to_numpy(dtype=np.float32)
    blocks_tr.append(np.nan_to_num(Xm_tr, nan=0.0, posinf=0.0, neginf=0.0))
    blocks_te.append(np.nan_to_num(Xm_te, nan=0.0, posinf=0.0, neginf=0.0))
    names += [f"mordred.{c}" for c in common_m]

    # Jazzy pose (6)
    with psycopg2.connect(**DB_PARAMS) as conn:
        pose_jazzy_df = pd.read_sql(
            "SELECT compound_id, sdc, sdx, sa, dga, dgp, dgtot "
            "FROM compound_boltz2_jazzy",
            conn,
        ).set_index("compound_id")
    jz_cols = list(JAZZY_FEATURE_COLS)
    Xjp_tr = pose_jazzy_df.reindex(train_ids)[jz_cols].to_numpy(dtype=np.float32)
    Xjp_te = pose_jazzy_df.reindex(test_ids)[jz_cols].to_numpy(dtype=np.float32)
    blocks_tr.append(np.nan_to_num(Xjp_tr, nan=0.0, posinf=0.0, neginf=0.0))
    blocks_te.append(np.nan_to_num(Xjp_te, nan=0.0, posinf=0.0, neginf=0.0))
    names += [f"jazzy_pose.{c}" for c in jz_cols]

    # Jazzy self (6)
    jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
    jazzy_test = load_jazzy(test_ids).reindex(index=test_ids)
    Xjs_tr = jazzy_train[jz_cols].to_numpy(dtype=np.float32)
    Xjs_te = jazzy_test[jz_cols].to_numpy(dtype=np.float32)
    blocks_tr.append(np.nan_to_num(Xjs_tr, nan=0.0, posinf=0.0, neginf=0.0))
    blocks_te.append(np.nan_to_num(Xjs_te, nan=0.0, posinf=0.0, neginf=0.0))
    names += [f"jazzy_self.{c}" for c in jz_cols]

    # RDKit full (217)
    rdkit_train = load_rdkit_full(train_ids)
    rdkit_test = load_rdkit_full(test_ids)
    common_r = sorted(rdkit_train.columns.intersection(rdkit_test.columns))
    Xr_tr = rdkit_train.loc[train_ids, common_r].to_numpy(dtype=np.float32)
    Xr_te = rdkit_test.loc[test_ids, common_r].to_numpy(dtype=np.float32)
    blocks_tr.append(np.nan_to_num(Xr_tr, nan=0.0, posinf=0.0, neginf=0.0))
    blocks_te.append(np.nan_to_num(Xr_te, nan=0.0, posinf=0.0, neginf=0.0))
    names += [f"rdkit.{c}" for c in common_r]

    # Boltz tier-0 scalars (17 + 2 derived = 19)
    col_sql = ", ".join(f"b.{c}" for c in BOLTZ_TIER0_COLS)
    with psycopg2.connect(**DB_PARAMS) as conn:
        boltz_df = pd.read_sql(
            f"""
            SELECT c.id AS compound_id, {col_sql},
                   (b.affinity_pred_value_1 - b.affinity_pred_value_2)
                       AS ensemble_diff_affinity,
                   (b.affinity_probability_binary_1
                      - b.affinity_probability_binary_2)
                       AS ensemble_diff_prob
            FROM compounds c
            LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
            """,
            conn,
        ).set_index("compound_id")
    tier0_cols = list(BOLTZ_TIER0_COLS) + [
        "ensemble_diff_affinity",
        "ensemble_diff_prob",
    ]
    Xt0_tr = boltz_df.reindex(train_ids)[tier0_cols].to_numpy(dtype=np.float32)
    Xt0_te = boltz_df.reindex(test_ids)[tier0_cols].to_numpy(dtype=np.float32)
    blocks_tr.append(np.nan_to_num(Xt0_tr, nan=0.0, posinf=0.0, neginf=0.0))
    blocks_te.append(np.nan_to_num(Xt0_te, nan=0.0, posinf=0.0, neginf=0.0))
    names += [f"boltz_tier0.{c}" for c in tier0_cols]

    # Boltz-2 tier-1 confidence aggregates (44) from parquet
    tier1_path = REPO_ROOT.joinpath("data", "boltz2_confidence_features.parquet")
    tier1_df = pd.read_parquet(tier1_path)
    Xt1_tr = tier1_df.reindex(train_ids).to_numpy(dtype=np.float32)
    Xt1_te = tier1_df.reindex(test_ids).to_numpy(dtype=np.float32)
    blocks_tr.append(np.nan_to_num(Xt1_tr, nan=0.0, posinf=0.0, neginf=0.0))
    blocks_te.append(np.nan_to_num(Xt1_te, nan=0.0, posinf=0.0, neginf=0.0))
    names += [f"boltz_tier1.{c}" for c in tier1_df.columns.tolist()]

    # 3D ligand (scalar 11 + 7 vector families + electroshape 15 + mordred3d)
    with psycopg2.connect(**DB_PARAMS) as conn:
        scalar_df = pd.read_sql(
            f"SELECT compound_id, {', '.join(D3_SCALAR_COLS)} "
            "FROM compound_boltz2_desc3d",
            conn,
        ).set_index("compound_id")
        vec_df = pd.read_sql(
            "SELECT compound_id, autocorr3d, getaway, morse, rdf, whim, "
            "usr, usrcat FROM compound_boltz2_desc3d_vector",
            conn,
        ).set_index("compound_id")
        skfp_df = pd.read_sql(
            "SELECT compound_id, electroshape FROM compound_boltz2_skfp3d",
            conn,
        ).set_index("compound_id")
        mord_df = pd.read_sql(
            "SELECT compound_id, descriptors FROM compound_boltz2_mordred3d",
            conn,
        ).set_index("compound_id")

    def _build3d(ids):
        s = scalar_df.reindex(ids)[list(D3_SCALAR_COLS)].to_numpy(dtype=np.float32)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        parts = [s]
        for col, dim in D3_VECTOR_SPEC:
            parts.append(_fill_vec(vec_df.reindex(ids)[col], dim))
        es = _fill_vec(skfp_df.reindex(ids)["electroshape"], 15)
        parts.append(es)
        mord, mord_cols = _mord_matrix(mord_df.reindex(ids)["descriptors"])
        parts.append(mord)
        return np.concatenate(parts, axis=1), mord_cols

    X3d_tr, mord3d_cols = _build3d(train_ids)
    X3d_te, _ = _build3d(test_ids)
    blocks_tr.append(X3d_tr)
    blocks_te.append(X3d_te)
    names += [f"d3_scalar.{c}" for c in D3_SCALAR_COLS]
    for col, dim in D3_VECTOR_SPEC:
        names += [f"d3_{col}.{i}" for i in range(dim)]
    names += [f"d3_electroshape.{i}" for i in range(15)]
    names += [f"d3_mordred3d.{c}" for c in mord3d_cols]

    X_tr = np.concatenate(blocks_tr, axis=1)
    X_te = np.concatenate(blocks_te, axis=1)
    assert X_tr.shape[1] == len(names)
    assert X_te.shape[1] == len(names)
    return X_tr, X_te, _sanitize(names)


def main() -> None:
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    assert len(train_ids) == len(y)

    print(f"Building matrix: train={len(train_ids)} test={len(test_ids)}")
    X_tr, X_te, feat_names = build_matrix(train_ids, test_ids)
    print(f"  Feature matrix: {X_tr.shape} (train) / {X_te.shape} (test)")

    # 5-fold UMAP CV, default params
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=5,
        n_clusters=50,
        seed=42,
    )

    oof = np.zeros_like(y)
    gain_matrix = np.zeros((5, len(feat_names)), dtype=np.float64)
    fold_maes: list[float] = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feat_names)
        dva = lgb.Dataset(Xva, label=yva, feature_name=feat_names, reference=dtr)
        model = lgb.train(
            LGBM_PARAMS,
            dtr,
            num_boost_round=5000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        preds = model.predict(Xva, num_iteration=model.best_iteration)
        oof[va_idx] = preds
        mae = float(np.mean(np.abs(yva - preds)))
        fold_maes.append(mae)
        gain_matrix[k] = model.feature_importance(importance_type="gain")
        print(f"  fold {k}: best_iter={model.best_iteration:>5d} val_MAE={mae:.4f}")

    oof_mae = float(np.mean(np.abs(y - oof)))
    baseline = float(np.mean(np.abs(y - y.mean())))
    oof_rae = float(np.sum(np.abs(y - oof)) / np.sum(np.abs(y - y.mean())))
    print(f"\nOOF MAE={oof_mae:.4f}  RAE={oof_rae:.4f}  baseline_MAE={baseline:.4f}")

    # Aggregate importance
    gain_mean = gain_matrix.mean(axis=0)
    gain_std = gain_matrix.std(axis=0)
    gain_nonzero_folds = (gain_matrix > 0).sum(axis=0)
    family = [n.split(".")[0] for n in feat_names]
    imp_df = pd.DataFrame(
        dict(
            feature=feat_names,
            family=family,
            gain_mean=gain_mean,
            gain_std=gain_std,
            nonzero_folds=gain_nonzero_folds,
        )
    )
    imp_df = imp_df.sort_values("gain_mean", ascending=False).reset_index(drop=True)
    imp_df["rank"] = np.arange(1, len(imp_df) + 1)
    imp_df["gain_share_pct"] = imp_df["gain_mean"] / imp_df["gain_mean"].sum() * 100

    csv_path = REPORT_DIR.joinpath("importance_all_desc.csv")
    imp_df.to_csv(csv_path, index=False)

    # Per-family summary
    fam_agg = (
        imp_df.groupby("family")
        .agg(
            n_features=("feature", "count"),
            gain_sum=("gain_mean", "sum"),
            gain_max=("gain_mean", "max"),
            nonzero=("nonzero_folds", lambda s: int((s >= 3).sum())),
        )
        .sort_values("gain_sum", ascending=False)
    )
    total_gain = imp_df["gain_mean"].sum()
    fam_agg["gain_share_pct"] = (fam_agg["gain_sum"] / total_gain * 100).round(2)
    fam_agg["gain_per_feat"] = (fam_agg["gain_sum"] / fam_agg["n_features"]).round(1)

    print("\n=== Family summary (rows where nonzero_folds>=3 count as 'nonzero') ===")
    print(fam_agg.to_string())

    print("\n=== Top 50 features ===")
    print(
        imp_df[["rank", "feature", "gain_mean", "nonzero_folds", "gain_share_pct"]]
        .head(50)
        .to_string(index=False)
    )

    log_path = REPORT_DIR.joinpath("importance_all_desc.log")
    with log_path.open("w") as f:
        f.write(f"Feature matrix: {X_tr.shape}\n")
        f.write(f"Fold MAEs: {fold_maes}\n")
        f.write(f"OOF MAE={oof_mae:.4f} RAE={oof_rae:.4f}\n\n")
        f.write("Family summary:\n")
        f.write(fam_agg.to_string())
        f.write("\n\nTop 50 features:\n")
        f.write(
            imp_df[["rank", "feature", "gain_mean", "nonzero_folds"]]
            .head(50)
            .to_string(index=False)
        )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
