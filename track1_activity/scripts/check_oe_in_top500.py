"""Inspect which OE features survive the per-fold LGBM-gain top-500 cut
for oe_cheme_2d_full_boltz_log2fc_pred (2126d -> 500).
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402
import run_train  # noqa: E402


def main() -> None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        tr_df = pd.read_sql(
            "SELECT t.id AS compound_id, t.pec50, c.std_smiles AS smiles "
            "FROM train_activity t JOIN compounds c ON c.id = t.compound_id ORDER BY t.id",
            conn,
        )
        te_df = pd.read_sql(
            "SELECT t.id AS compound_id, c.std_smiles AS smiles "
            "FROM test_activity t JOIN compounds c ON c.id = t.compound_id ORDER BY t.id",
            conn,
        )

    X_tr, _ = run_train.load_features(
        "oe_cheme_2d_full_boltz_log2fc_pred", tr_df, te_df
    )
    y = tr_df["pec50"].to_numpy(dtype=np.float32)
    col_mean = np.nanmean(X_tr, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_tr = np.where(np.isfinite(X_tr), X_tr, col_mean).astype(np.float32)

    # Feature labels
    # oemedchem 15 + rocs 6 + quacpac 2 + cheme 300 + 2d_full_boltz_log2fc_pred 1803
    labels = (
        [
            f"oe_{n}"
            for n in [
                "xlogp",
                "psa_2d",
                "mw",
                "hba",
                "hbd",
                "lipinski_hba",
                "lipinski_hbd",
                "aromatic_ring_count",
                "rotatable_bond_count",
                "fraction_csp3",
                "halide_fraction",
                "longest_unbranched_c_chain",
                "longest_unbranched_heavy_chain",
                "num_unspecified_atom_stereo",
                "num_unspecified_bond_stereo",
            ]
        ]
        + [
            f"rocs_{n}"
            for n in [
                "max_shape_tanimoto",
                "max_color_tanimoto",
                "max_combo_tanimoto",
                "mean_shape_tanimoto",
                "mean_color_tanimoto",
                "mean_combo_tanimoto",
            ]
        ]
        + ["quacpac_formal_charge", "quacpac_n_tautomers"]
        + [f"cheme_{i}" for i in range(300)]
        + [f"mord_{i}" for i in range(1515)]
        + [f"jazzy_{i}" for i in range(6)]
        + [f"rdkit_{i}" for i in range(217)]
        + [f"tier0_{i}" for i in range(19)]
        + [f"tier1_{i}" for i in range(44)]
        + ["log2fc_pred_0", "log2fc_pred_1"]
    )
    assert len(labels) == X_tr.shape[1]

    splits = umap_split_indices(
        tr_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )

    # Per-fold selection
    per_fold_sel: list[set[int]] = []
    per_fold_gain = np.zeros(X_tr.shape[1])
    for fold, (tr_idx, _va_idx) in enumerate(splits):
        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_tr[tr_idx], y[tr_idx])
        gain = lgbm.booster_.feature_importance(importance_type="gain")
        sel = set(np.argsort(-gain)[:500].tolist())
        per_fold_sel.append(sel)
        per_fold_gain += gain

    # Survival of new OE features (first 23 indices)
    new_feat_indices = list(range(23))  # oemedchem 0-14, rocs 15-20, quacpac 21-22

    print("OE feature per-fold survival in top500 (out of 5 folds):")
    print(f"  {'feature':<42s}  {'survived':<10s}  {'total gain':<12s}")
    print("  " + "-" * 70)
    for idx in new_feat_indices:
        n_survived = sum(1 for s in per_fold_sel if idx in s)
        print(f"  {labels[idx]:<42s}  {n_survived}/5        {per_fold_gain[idx]:12.1f}")

    # Family-level survival summary
    print("\nFamily-level survival share in top500 (averaged across folds):")
    family_of_idx: dict[int, str] = {}
    for idx in range(23):
        if idx < 15:
            family_of_idx[idx] = "oemedchem"
        elif idx < 21:
            family_of_idx[idx] = "rocs"
        else:
            family_of_idx[idx] = "quacpac"
    for idx in range(23, 323):
        family_of_idx[idx] = "cheme"
    for idx in range(323, 323 + 1515):
        family_of_idx[idx] = "mordred"
    for idx in range(323 + 1515, 323 + 1515 + 6):
        family_of_idx[idx] = "pose_jazzy"
    for idx in range(323 + 1515 + 6, 323 + 1515 + 6 + 217):
        family_of_idx[idx] = "rdkit_full"
    for idx in range(323 + 1515 + 6 + 217, 323 + 1515 + 6 + 217 + 19):
        family_of_idx[idx] = "tier0"
    for idx in range(323 + 1515 + 6 + 217 + 19, 323 + 1515 + 6 + 217 + 19 + 44):
        family_of_idx[idx] = "tier1"
    for idx in range(323 + 1515 + 6 + 217 + 19 + 44, X_tr.shape[1]):
        family_of_idx[idx] = "log2fc_pred"

    fam_counts: dict[str, int] = {}
    fam_totals: dict[str, int] = {}
    for idx, fam in family_of_idx.items():
        fam_totals[fam] = fam_totals.get(fam, 0) + 1
        avg_survived = sum(1 for s in per_fold_sel if idx in s) / 5.0
        fam_counts[fam] = fam_counts.get(fam, 0.0) + avg_survived

    print(f"  {'family':<14s} {'n_feat':<8s} {'avg_picked':<12s} {'avg_share_%':<12s}")
    print("  " + "-" * 55)
    for fam in sorted(fam_counts, key=lambda f: -fam_counts[f]):
        share = fam_counts[fam] / fam_totals[fam] * 100
        print(
            f"  {fam:<14s} {fam_totals[fam]:<8d} "
            f"{fam_counts[fam]:<12.1f} {share:<12.1f}"
        )


if __name__ == "__main__":
    main()
