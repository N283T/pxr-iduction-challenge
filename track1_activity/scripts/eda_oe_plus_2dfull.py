"""LGBM importance for 2d_full_boltz + OE features (no chemeleon).

Feature set:
  - 2d_full_boltz (1817): Mordred 1531 + Pose-Jazzy 6 + RDKit 217 +
                          Boltz Tier-0 19 + Tier-1 44
  - oemedchem (15 after dropping constant anionic_carbon_count)
  - rocs (6)
  - quacpac (2: n_tautomers, formal_charge)
  Total: ~1840
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import lightgbm as lgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import run_train  # noqa: E402


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
    "num_unspecified_atom_stereo",
    "num_unspecified_bond_stereo",
]  # anionic_carbon_count dropped (constant 0 on PXR)

ROCS_COLS = [
    "max_shape_tanimoto",
    "max_color_tanimoto",
    "max_combo_tanimoto",
    "mean_shape_tanimoto",
    "mean_color_tanimoto",
    "mean_combo_tanimoto",
]

QUACPAC_COLS = ["n_tautomers", "formal_charge"]


def load_train_test_dfs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    conn = psycopg2.connect(**DB_PARAMS)
    tr = pd.read_sql(
        """SELECT t.compound_id AS id, c.std_smiles AS smiles, t.pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        conn,
    )
    te = pd.read_sql(
        """SELECT t.compound_id AS id, c.std_smiles AS smiles
           FROM test_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        conn,
    )
    conn.close()
    return tr, te, tr["pec50"].to_numpy(dtype=float), tr["smiles"].tolist()


def load_oe_features(compound_ids: list[int]) -> pd.DataFrame:
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
        "SELECT compound_id, n_tautomers FROM compound_tautomers "
        "WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    charge = pd.read_sql(
        "SELECT compound_id, formal_charge FROM compound_quacpac "
        "WHERE compound_id = ANY(%s)",
        conn,
        params=(compound_ids,),
    ).set_index("compound_id")
    conn.close()
    df = oe.join(rocs, how="left").join(taut, how="left").join(charge, how="left")
    return df.reindex(compound_ids)


def main() -> None:
    print("Loading train/test splits...")
    train_df, test_df, y, smiles_list = load_train_test_dfs()
    train_ids = train_df["id"].tolist()
    test_ids = test_df["id"].tolist()
    print(f"  train={len(train_ids)} test={len(test_ids)}")

    print("\nLoading 2d_full_boltz (Mordred+Jazzy+RDKit+Tier0+Tier1)...")
    X_2d_tr, X_2d_te = run_train.load_features("2d_full_boltz", train_df, test_df)
    print(f"  2d_full_boltz: {X_2d_tr.shape}")

    print("\nLoading OE features for train...")
    oe_df = load_oe_features(train_ids)
    oe_cols = OEMEDCHEM_COLS + ROCS_COLS + QUACPAC_COLS
    X_oe_tr = np.ascontiguousarray(oe_df[oe_cols].to_numpy(dtype=np.float32))
    col_means = np.nan_to_num(np.nanmean(X_oe_tr, axis=0), nan=0.0)
    nan_mask = np.isnan(X_oe_tr)
    if nan_mask.any():
        X_oe_tr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    print(f"  OE: {X_oe_tr.shape}")

    # Concatenate
    X_full = np.concatenate([X_2d_tr, X_oe_tr], axis=1)
    print(f"  Combined: {X_full.shape}")

    # Feature names: derive from 2d_full_boltz piece names plus oe cols.
    # Layout (per run_train.load_features verbose output):
    #   Mordred 1515 + Pose-Jazzy 6 + RDKit 217 + Tier-0 19 + Tier-1 44 = 1801
    n_jz = 6
    n_rd = 217
    n_t0 = 19
    n_t1 = 44
    n_mord = X_2d_tr.shape[1] - (n_jz + n_rd + n_t0 + n_t1)
    print(
        f"  (inferred layout: mord {n_mord} + jz {n_jz} + rd {n_rd} "
        f"+ t0 {n_t0} + t1 {n_t1} = {X_2d_tr.shape[1]})"
    )
    feature_names = (
        [f"mord_{i}" for i in range(n_mord)]
        + [f"pose_jazzy_{i}" for i in range(n_jz)]
        + [f"rdkit_{i}" for i in range(n_rd)]
        + [f"boltz_tier0_{i}" for i in range(n_t0)]
        + [f"boltz_tier1_{i}" for i in range(n_t1)]
        + oe_cols
    )
    assert len(feature_names) == X_full.shape[1]

    # UMAP split
    print("\nComputing UMAP 5-fold split (Morgan+Jaccard, k=50, seed=42)...")
    splits = umap_split_indices(smiles_list, n_splits=5, n_clusters=50, seed=42)
    fold_idx = np.zeros(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    # Evaluate 3 variants: 2d_full_boltz only, 2d_full_boltz + OE, and delta
    def cv(X: np.ndarray, tag: str) -> tuple[dict, np.ndarray]:
        oof = np.zeros_like(y)
        total_gain = np.zeros(X.shape[1])
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
        mae = float(np.mean(np.abs(oof - y)))
        rae = mae / float(np.mean(np.abs(y - y.mean())))
        sp = float(spearmanr(oof, y).statistic)
        r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
        print(
            f"  {tag:<35s}  n={X.shape[1]:<5d}  MAE={mae:.4f}  RAE={rae:.4f}  "
            f"R2={r2:+.3f}  Sp={sp:.4f}"
        )
        return {"mae": mae, "rae": rae, "sp": sp, "r2": float(r2)}, total_gain

    print("\nLGBM 5-fold OOF comparison:")
    _, _ = cv(X_2d_tr, "2d_full_boltz (no OE)")
    _, gain_full = cv(X_full, "2d_full_boltz + OE (15+6+2)")

    # Gain importance analysis on the combined model
    print("\n" + "=" * 80)
    print("LGBM gain importance: 2d_full_boltz + OE (combined)")
    print("=" * 80)

    total_g = gain_full.sum()

    def family(fname: str) -> str:
        if fname.startswith("mord_"):
            return "mordred"
        if fname.startswith("pose_jazzy_"):
            return "pose_jazzy"
        if fname.startswith("rdkit_"):
            return "rdkit_full"
        if fname.startswith("boltz_tier0_"):
            return "boltz_tier0"
        if fname.startswith("boltz_tier1_"):
            return "boltz_tier1"
        if fname in OEMEDCHEM_COLS:
            return "oemedchem"
        if fname in ROCS_COLS:
            return "rocs"
        if fname in QUACPAC_COLS:
            return "quacpac"
        return "other"

    df_imp = pd.DataFrame({"feature": feature_names, "gain": gain_full})
    df_imp["family"] = df_imp["feature"].map(family)
    df_imp["gain_share_%"] = (df_imp["gain"] / total_g * 100).round(3)

    print("\nGain share by family:")
    fam_agg = (
        df_imp.groupby("family")
        .agg(n=("feature", "size"), gain=("gain", "sum"))
        .sort_values("gain", ascending=False)
    )
    fam_agg["share_%"] = (fam_agg["gain"] / total_g * 100).round(2)
    print(fam_agg.to_string())

    print("\nTop 30 features overall:")
    print(df_imp.sort_values("gain", ascending=False).head(30).to_string(index=False))

    # Specifically: where do OE features rank?
    print("\nOE feature ranking within combined model:")
    oe_rows = df_imp[df_imp["family"].isin(["oemedchem", "rocs", "quacpac"])]
    oe_rows = oe_rows.sort_values("gain", ascending=False)
    # Determine overall rank
    df_imp_sorted = df_imp.sort_values("gain", ascending=False).reset_index(drop=True)
    rank_map = {f: i + 1 for i, f in enumerate(df_imp_sorted["feature"])}
    oe_rows = oe_rows.copy()
    oe_rows["overall_rank"] = oe_rows["feature"].map(rank_map)
    print(
        oe_rows[
            ["family", "feature", "gain", "gain_share_%", "overall_rank"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
