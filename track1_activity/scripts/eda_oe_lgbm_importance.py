"""LGBM gain importance for the E_all_combined feature set (241 features)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import lightgbm as lgb  # noqa: E402


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


def load_all() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    conn = psycopg2.connect(**DB_PARAMS)
    df_tgt = pd.read_sql(
        """SELECT t.compound_id, c.std_smiles AS smiles, t.pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        conn,
    )
    compound_ids = df_tgt["compound_id"].tolist()
    smiles_list = df_tgt["smiles"].tolist()
    y = df_tgt["pec50"].to_numpy(dtype=float)

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

    rdkit_df = pd.json_normalize(rdkit["descriptors"]).set_index(rdkit.index)
    rdkit_df.columns = [f"rdkit_{c}" for c in rdkit_df.columns]

    df = oe.join(rocs, how="left").join(taut, how="left").join(rdkit_df, how="left")
    df = df.reindex(compound_ids)
    return df, y, smiles_list


def main() -> None:
    print("Loading E_all_combined (241 features)...")
    df, y, smiles_list = load_all()
    # Drop constant columns (anionic_carbon_count is all 0 on PXR)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        print(f"  dropping {len(const_cols)} constant columns: {const_cols}")
        df = df.drop(columns=const_cols)

    print(f"  shape: {df.shape}")
    print("Computing UMAP 5-fold split...")
    splits = umap_split_indices(smiles_list, n_splits=5, n_clusters=50, seed=42)
    fold_idx = np.zeros(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    feature_names = list(df.columns)
    X = np.ascontiguousarray(df.to_numpy(dtype=float))
    col_means = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Train per-fold models, accumulate gain importance
    total_gain = np.zeros(X.shape[1])
    total_split = np.zeros(X.shape[1])
    oof = np.zeros_like(y)
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
        total_split += m.booster_.feature_importance(importance_type="split")

    mae = float(np.mean(np.abs(oof - y)))
    from scipy.stats import spearmanr

    sp = float(spearmanr(oof, y).statistic)
    print(f"\nOOF MAE={mae:.4f}  Spearman={sp:.4f}")

    # Group importance by family
    def family(fname: str) -> str:
        if fname in OEMEDCHEM_COLS:
            return "oemedchem"
        if fname in ROCS_COLS:
            return "rocs"
        if fname in ("n_tautomers", "formal_charge"):
            return "quacpac"
        if fname.startswith("rdkit_"):
            return "rdkit_full"
        return "other"

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": total_gain,
            "split": total_split,
        }
    )
    df_imp["family"] = df_imp["feature"].map(family)

    total_g = df_imp["gain"].sum()
    print(f"\nTotal gain: {total_g:.0f}")

    print("\nGain share by family:")
    fam_agg = (
        df_imp.groupby("family")
        .agg(n=("feature", "size"), gain=("gain", "sum"))
        .sort_values("gain", ascending=False)
    )
    fam_agg["share_%"] = (fam_agg["gain"] / total_g * 100).round(2)
    print(fam_agg.to_string())

    print("\nTop 30 features by gain:")
    df_imp_sorted = df_imp.sort_values("gain", ascending=False).head(30)
    df_imp_sorted["gain_share_%"] = (df_imp_sorted["gain"] / total_g * 100).round(2)
    print(
        df_imp_sorted[["family", "feature", "gain", "gain_share_%"]].to_string(
            index=False
        )
    )

    # New-feature (non-rdkit) specific ranking
    print("\nTop 20 NEW features (oemedchem + rocs + quacpac):")
    df_new = df_imp[df_imp["family"] != "rdkit_full"].sort_values(
        "gain", ascending=False
    )
    df_new["gain_share_%"] = (df_new["gain"] / total_g * 100).round(2)
    print(
        df_new[["family", "feature", "gain", "gain_share_%"]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
