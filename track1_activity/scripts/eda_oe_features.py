"""Exploratory EDA + LightGBM baseline for the new OpenEye features.

Evaluates four feature sets on the UMAP 5-fold split:
  A. oemedchem only (16 scalar)
  B. ROCS only (6 scalar summary)
  C. oemedchem + ROCS + tautomer_count (23 scalar)
  D. Full RDKit descriptors (217) — reference for gap comparison

Outputs per-feature target correlations and OOF MAE/RAE/Spearman per set.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import lightgbm as lgb  # noqa: E402


def load_train_ids_and_target() -> tuple[list[int], list[str], np.ndarray]:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        """SELECT t.compound_id, c.std_smiles AS smiles, t.pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        conn,
    )
    conn.close()
    return (
        df["compound_id"].tolist(),
        df["smiles"].tolist(),
        df["pec50"].to_numpy(dtype=float),
    )


OEMEDCHEM_COLS = [
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
    "anionic_carbon_count",
    "num_unspecified_atom_stereo",
    "num_unspecified_bond_stereo",
]

ROCS_COLS = [
    "max_shape_tanimoto",
    "max_color_tanimoto",
    "max_combo_tanimoto",
    "mean_shape_tanimoto",
    "mean_color_tanimoto",
    "mean_combo_tanimoto",
]


def load_features(compound_ids: list[int]) -> pd.DataFrame:
    """Join oemedchem + ROCS + tautomer_count for the given compound ids."""
    conn = psycopg2.connect(**DB_PARAMS)
    oe = pd.read_sql(
        f"SELECT compound_id, {','.join(OEMEDCHEM_COLS)} "
        f"FROM compound_oemedchem WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    rocs = pd.read_sql(
        f"SELECT compound_id, {','.join(ROCS_COLS)} "
        f"FROM compound_rocs WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    taut = pd.read_sql(
        "SELECT compound_id, n_tautomers, formal_charge "
        "FROM compound_tautomers t JOIN compound_quacpac q USING(compound_id) "
        "WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    rdkit = pd.read_sql(
        "SELECT compound_id, descriptors FROM compound_descriptors_full "
        "WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    conn.close()

    # Expand RDKit JSONB
    rdkit_df = pd.json_normalize(rdkit["descriptors"]).set_index(rdkit.index)
    rdkit_df.columns = [f"rdkit_{c}" for c in rdkit_df.columns]

    df = oe.join(rocs, how="left").join(taut, how="left").join(rdkit_df, how="left")
    return df.reindex(compound_ids)


def lgbm_cv(X: np.ndarray, y: np.ndarray, fold_idx: np.ndarray, name: str) -> dict:
    """5-fold CV with LightGBM default hyperparameters."""
    oof = np.zeros_like(y, dtype=float)
    for fold in range(fold_idx.max() + 1):
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

    mae = float(np.mean(np.abs(oof - y)))
    rae = mae / float(np.mean(np.abs(y - y.mean())))
    spearman = float(spearmanr(oof, y).statistic)
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)

    return {
        "name": name,
        "n_features": X.shape[1],
        "MAE": mae,
        "RAE": rae,
        "R2": float(r2),
        "Spearman": spearman,
    }


def main() -> None:
    print("Loading train pEC50 + compound ids...")
    compound_ids, smiles_list, y = load_train_ids_and_target()
    print(f"  {len(compound_ids)} train compounds")

    print("Loading OE features...")
    t0 = time.time()
    df = load_features(compound_ids)
    print(f"  loaded {df.shape} in {time.time() - t0:.1f}s")

    # Coverage
    oe_cov = df[OEMEDCHEM_COLS].notna().all(axis=1).mean()
    rocs_cov = df[ROCS_COLS].notna().all(axis=1).mean()
    print(f"  coverage: oemedchem {oe_cov:.3f}  rocs {rocs_cov:.3f}")

    # Correlations with target
    print("\nPer-feature correlation with pEC50 (Spearman, |top 20|):")
    corrs = []
    for col in OEMEDCHEM_COLS + ROCS_COLS + ["n_tautomers", "formal_charge"]:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() < 100:
            continue
        sp = spearmanr(vals[mask], y[mask]).statistic
        corrs.append((col, sp))
    corrs.sort(key=lambda r: abs(r[1]), reverse=True)
    for col, sp in corrs[:20]:
        print(f"  {col:<35s}  Spearman={sp:+.3f}")

    # Compute UMAP split
    print("\nComputing UMAP 5-fold split (Morgan+Jaccard, k=50, seed=42)...")
    splits = umap_split_indices(smiles_list, n_splits=5, n_clusters=50, seed=42)
    fold_idx = np.zeros(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    # Feature sets to evaluate
    feature_sets = {
        "A_oemedchem_only": OEMEDCHEM_COLS,
        "B_rocs_only": ROCS_COLS,
        "C_oemedchem+rocs+taut": (
            OEMEDCHEM_COLS + ROCS_COLS + ["n_tautomers", "formal_charge"]
        ),
        "D_rdkit_full_217": [c for c in df.columns if c.startswith("rdkit_")],
        "E_all_combined": (
            OEMEDCHEM_COLS
            + ROCS_COLS
            + ["n_tautomers", "formal_charge"]
            + [c for c in df.columns if c.startswith("rdkit_")]
        ),
    }

    print("\nLGBM 5-fold OOF (UMAP split, seed=42):")
    print(
        f"  {'set':<25s} {'n_feat':<8s} {'MAE':<8s} {'RAE':<8s} {'R2':<8s} {'Sp':<8s}"
    )
    print("  " + "-" * 70)
    for name, cols in feature_sets.items():
        use_cols = [c for c in cols if c in df.columns]
        X = np.ascontiguousarray(df[use_cols].to_numpy(dtype=float))
        # Fill NaN with column mean (LGBM can handle NaN but keep consistent)
        col_means = np.nanmean(X, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        m = lgbm_cv(X, y, fold_idx, name)
        print(
            f"  {m['name']:<25s} {m['n_features']:<8d} "
            f"{m['MAE']:.4f}  {m['RAE']:.4f}  {m['R2']:+.3f}  {m['Spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
