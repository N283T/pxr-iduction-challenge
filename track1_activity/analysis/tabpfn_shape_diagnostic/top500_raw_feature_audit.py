#!/usr/bin/env python
"""Audit raw features selected by the cheme 2D/Boltz/log2fc top-500 selector."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import (  # noqa: E402
    JAZZY_FEATURE_COLS,
    load_mordred,
    load_rdkit_full,
    load_test_smiles,
    load_train_mordred,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402

OUT_DIR = (
    Path(__file__).resolve().parent.joinpath("outputs", "top500_raw_feature_audit")
)
FEATURE = "cheme_2d_full_boltz_log2fc_pred_seed10ens"
K = 500
SEED = 42


def family_of(feature: str) -> str:
    if feature.startswith("chemeleon_"):
        return "chemeleon"
    if feature.startswith("mordred__"):
        return "mordred"
    if feature.startswith("rdkit__"):
        return "rdkit_full"
    if feature.startswith("pose_jazzy__"):
        return "pose_jazzy"
    if feature.startswith("boltz_tier0__"):
        return "boltz_tier0"
    if feature.startswith("boltz_tier1__"):
        return "boltz_tier1_conf"
    if feature.startswith("log2fc_pred__"):
        return "log2fc_pred"
    return "unknown"


def build_feature_names() -> list[str]:
    train_ids = run_train.load_compound_ids("train")
    test_ids = run_train.load_compound_ids("test")

    names: list[str] = [f"chemeleon_{i:03d}" for i in range(300)]

    mordred_train, _ = load_train_mordred()
    mordred_test = load_mordred(test_ids)
    common_m = mordred_train.columns.intersection(mordred_test.columns)
    names.extend([f"mordred__{c}" for c in common_m])

    names.extend([f"pose_jazzy__{c}" for c in JAZZY_FEATURE_COLS])

    rdkit_train = load_rdkit_full(train_ids)
    rdkit_test = load_rdkit_full(test_ids)
    common_r = rdkit_train.columns.intersection(rdkit_test.columns)
    names.extend([f"rdkit__{c}" for c in common_r])

    boltz2_cols = [
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
        "ensemble_diff_affinity",
        "ensemble_diff_prob",
    ]
    names.extend([f"boltz_tier0__{c}" for c in boltz2_cols])

    tier1_path = REPO_ROOT.joinpath("data", "boltz2_confidence_features.parquet")
    tier1_cols = list(pd.read_parquet(tier1_path).columns)
    names.extend([f"boltz_tier1__{c}" for c in tier1_cols])

    names.extend(["log2fc_pred__log2fc_8p25_pred", "log2fc_pred__log2fc_33_pred"])
    return names


def load_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    X_train, _ = run_train.load_features(FEATURE, train_df, test_df)
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    names = build_feature_names()
    if len(names) != X_train.shape[1]:
        raise RuntimeError(f"feature name mismatch: {len(names)} vs {X_train.shape[1]}")
    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, col_mean).astype(np.float32)
    return train_df, X_train, y, names


def main() -> None:
    train_df, X, y, names = load_matrix()
    folds = umap_split_indices(
        train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=SEED
    )

    gain_matrix = np.zeros((len(folds), X.shape[1]), dtype=np.float64)
    selected_matrix = np.zeros((len(folds), X.shape[1]), dtype=bool)
    family_rows = []
    top_rows = []

    for fold, (tr_idx, _va_idx) in enumerate(folds):
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=SEED,
            verbose=-1,
        )
        model.fit(X[tr_idx], y[tr_idx])
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_matrix[fold] = gain
        sel = np.argsort(-gain)[:K]
        selected_matrix[fold, sel] = True

        fold_df = pd.DataFrame(
            {
                "feature": np.asarray(names)[sel],
                "family": [family_of(names[i]) for i in sel],
                "gain": gain[sel],
                "fold": fold,
                "rank": np.arange(1, len(sel) + 1),
            }
        )
        top_rows.append(fold_df)
        family_rows.append(
            fold_df.groupby("family", as_index=False)
            .agg(n_selected=("feature", "size"), gain_sum=("gain", "sum"))
            .assign(fold=fold)
        )

    selected_count = selected_matrix.sum(axis=0)
    gain_mean = gain_matrix.mean(axis=0)
    gain_std = gain_matrix.std(axis=0)
    out = pd.DataFrame(
        {
            "feature": names,
            "family": [family_of(n) for n in names],
            "selected_folds": selected_count,
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_nonzero_folds": (gain_matrix > 0).sum(axis=0),
        }
    ).sort_values(["gain_mean", "selected_folds"], ascending=False)
    out["gain_share_pct"] = out["gain_mean"] / out["gain_mean"].sum() * 100.0

    top_fold_df = pd.concat(top_rows, ignore_index=True)
    family_fold = pd.concat(family_rows, ignore_index=True)
    family_mean = (
        family_fold.groupby("family", as_index=False)
        .agg(
            selected_mean=("n_selected", "mean"),
            selected_min=("n_selected", "min"),
            selected_max=("n_selected", "max"),
            gain_mean=("gain_sum", "mean"),
        )
        .sort_values("gain_mean", ascending=False)
    )
    family_mean["gain_share_pct"] = (
        family_mean["gain_mean"] / family_mean["gain_mean"].sum() * 100.0
    )

    stable = out[out["selected_folds"].eq(len(folds))].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR.joinpath("feature_gain_summary.csv"), index=False)
    top_fold_df.to_csv(OUT_DIR.joinpath("selected_features_by_fold.csv"), index=False)
    family_mean.to_csv(OUT_DIR.joinpath("family_summary.csv"), index=False)

    report = [
        "# Top500 Raw Feature Audit",
        "",
        f"Feature set: `{FEATURE}`",
        f"Raw feature count: `{X.shape[1]}`",
        f"Selection: per-fold LGBM gain top-{K}, matching the top500 TabPFN pipeline.",
        "",
        "## Family Composition",
        "",
        family_mean.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Top Features By Mean Gain",
        "",
        out.head(40)[
            [
                "feature",
                "family",
                "selected_folds",
                "gain_mean",
                "gain_share_pct",
                "gain_nonzero_folds",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Stable Selected Features",
        "",
        f"Selected in all 5 folds: `{len(stable)}`",
        "",
        stable.head(60)[
            ["feature", "family", "gain_mean", "gain_share_pct", "gain_nonzero_folds"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    OUT_DIR.joinpath("report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote top500 raw feature audit outputs to {OUT_DIR}")
    print(family_mean.to_string(index=False))
    print("\nTop 20:")
    print(
        out.head(20)[
            ["feature", "family", "selected_folds", "gain_share_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
