#!/usr/bin/env -S pixi run python
"""Build cross-fit dose-response latent features from internal assay labels.

The latent target intentionally excludes PXR pEC50. For train rows, latent
features are predicted out-of-fold by a chemistry-to-latent surrogate whose
training set excludes the validation fold's auxiliary assay labels. Test rows
are predicted by a final surrogate trained on all internally observed assay
rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_rdkit_full  # noqa: E402
from features import FP_REGISTRY, smiles_to_mols  # noqa: E402
from splits import umap_split_indices  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data")
DEFAULT_OUT = OUT_DIR.joinpath("dose_response_latent.parquet")
DEFAULT_META = OUT_DIR.joinpath("dose_response_latent_meta.json")

ASSAY_BASE_COLUMNS = [
    "pxr_emax",
    "pxr_emax_vs_pos_ctrl",
    "counter_present",
    "counter_pec50",
    "counter_emax",
    "counter_emax_vs_pos_ctrl",
    "log2fc_8p25",
    "log2fc_33",
    "log2fc_99",
]

ASSAY_DERIVED_COLUMNS = [
    "log2fc_slope_8p25_33",
    "log2fc_slope_33_99",
    "log2fc_max",
    "log2fc_auc",
]


def build_assay_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return internal assay-shape matrix without the pEC50 target column."""
    work = pd.DataFrame(index=df.index)
    for col in ASSAY_BASE_COLUMNS:
        work[col] = df[col] if col in df.columns else np.nan

    work["log2fc_slope_8p25_33"] = work["log2fc_33"] - work["log2fc_8p25"]
    work["log2fc_slope_33_99"] = work["log2fc_99"] - work["log2fc_33"]
    work["log2fc_max"] = work[["log2fc_8p25", "log2fc_33", "log2fc_99"]].max(
        axis=1, skipna=True
    )
    work["log2fc_auc"] = work[["log2fc_8p25", "log2fc_33", "log2fc_99"]].mean(
        axis=1, skipna=True
    )
    return work[ASSAY_BASE_COLUMNS + ASSAY_DERIVED_COLUMNS]


def latent_training_mask(
    compound_ids: np.ndarray,
    assay_counts: np.ndarray,
    validation_ids: set[int],
    min_observed: int,
) -> np.ndarray:
    """Select rows with enough auxiliary labels, excluding validation IDs."""
    in_validation = np.isin(compound_ids, np.asarray(list(validation_ids), dtype=int))
    return (assay_counts >= min_observed) & ~in_validation


def _load_assay_frame() -> pd.DataFrame:
    sql = """
    WITH sc AS (
      SELECT compound_id,
        AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
                 THEN log2_fc_estimate END) AS log2fc_8p25,
        AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
                 THEN log2_fc_estimate END) AS log2fc_33,
        AVG(CASE WHEN concentration_m BETWEEN 9.85e-5 AND 9.95e-5
                 THEN log2_fc_estimate END) AS log2fc_99
      FROM single_concentration
      GROUP BY compound_id
    )
    SELECT
      c.id AS compound_id,
      c.std_smiles AS smiles,
      t.emax_estimate AS pxr_emax,
      t.emax_vs_pos_ctrl AS pxr_emax_vs_pos_ctrl,
      CASE WHEN ca.compound_id IS NULL THEN NULL ELSE 1.0 END AS counter_present,
      ca.pec50 AS counter_pec50,
      ca.emax_estimate AS counter_emax,
      ca.emax_vs_pos_ctrl AS counter_emax_vs_pos_ctrl,
      sc.log2fc_8p25,
      sc.log2fc_33,
      sc.log2fc_99,
      CASE WHEN t.compound_id IS NULL THEN 0 ELSE 1 END AS is_train,
      CASE WHEN te.compound_id IS NULL THEN 0 ELSE 1 END AS is_test
    FROM compounds c
    LEFT JOIN train_activity t ON t.compound_id = c.id
    LEFT JOIN test_activity te ON te.compound_id = c.id
    LEFT JOIN counter_assay ca ON ca.compound_id = c.id
    LEFT JOIN sc ON sc.compound_id = c.id
    WHERE c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def _load_train_order_ids() -> list[int]:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
        return [int(row[0]) for row in cur.fetchall()]


def _build_chemistry_features(df: pd.DataFrame) -> np.ndarray:
    compound_ids = df["compound_id"].astype(int).tolist()
    desc = load_rdkit_full(compound_ids).reindex(compound_ids)
    desc_values = desc.to_numpy(dtype=np.float32)
    desc_values = np.nan_to_num(desc_values, nan=0.0, posinf=0.0, neginf=0.0)

    mols = smiles_to_mols(df["smiles"].tolist())
    morgan = FP_REGISTRY["count_morgan_r2_2048"](mols).astype(np.float32)
    return np.concatenate([desc_values, morgan], axis=1).astype(np.float32)


def _fit_latent(assay_values: np.ndarray, n_components: int, seed: int):
    n_components = min(n_components, assay_values.shape[1], assay_values.shape[0] - 1)
    if n_components < 1:
        raise ValueError("Need at least two assay rows to fit latent PCA")
    pipe = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
    )
    latent = pipe.fit_transform(assay_values).astype(np.float32)
    return pipe, latent


def _surrogate(seed: int, n_estimators: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        min_samples_leaf=3,
        max_features=0.35,
        n_jobs=-1,
    )


def build_latent_predictions(
    df: pd.DataFrame,
    X_chem: np.ndarray,
    n_components: int,
    min_observed: int,
    n_estimators: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    compound_ids = df["compound_id"].to_numpy(dtype=int)
    assay = build_assay_matrix(df)
    assay_values = assay.to_numpy(dtype=np.float32)
    assay_counts = np.isfinite(assay_values).sum(axis=1)
    train_ids = _load_train_order_ids()
    train_id_set = set(train_ids)
    test_ids = df.loc[df["is_test"].eq(1), "compound_id"].astype(int).tolist()

    train_positions = np.asarray(
        [i for i, cid in enumerate(compound_ids) if int(cid) in train_id_set],
        dtype=int,
    )
    train_smiles = df.iloc[train_positions]["smiles"].tolist()
    folds = umap_split_indices(train_smiles, n_splits=5, seed=42, n_clusters=50)
    if len(folds) != 5:
        raise ValueError(f"Expected 5 UMAP folds, got {len(folds)}")

    oof = np.full((len(compound_ids), n_components), np.nan, dtype=np.float32)
    base_model = _surrogate(seed=seed, n_estimators=n_estimators)
    fold_stats = []

    for fold_idx, (_tr_idx, va_idx) in enumerate(folds):
        validation_ids = set(compound_ids[train_positions[va_idx]].astype(int))
        fit_mask = latent_training_mask(
            compound_ids=compound_ids,
            assay_counts=assay_counts,
            validation_ids=validation_ids,
            min_observed=min_observed,
        )
        latent_pipe, latent_y = _fit_latent(
            assay_values[fit_mask], n_components=n_components, seed=seed + fold_idx
        )
        model = clone(base_model)
        model.set_params(random_state=seed + fold_idx)
        model.fit(X_chem[fit_mask], latent_y)
        pred = model.predict(X_chem[train_positions[va_idx]]).astype(np.float32)
        oof[train_positions[va_idx], : pred.shape[1]] = pred
        fold_stats.append(
            {
                "fold": fold_idx,
                "latent_train_rows": int(fit_mask.sum()),
                "validation_rows": int(len(va_idx)),
                "explained_variance": [
                    float(v)
                    for v in latent_pipe.named_steps["pca"].explained_variance_ratio_
                ],
            }
        )

    full_mask = assay_counts >= min_observed
    full_pipe, full_latent_y = _fit_latent(
        assay_values[full_mask], n_components=n_components, seed=seed
    )
    full_model = clone(base_model)
    full_model.fit(X_chem[full_mask], full_latent_y)
    full_pred = full_model.predict(X_chem).astype(np.float32)

    out = pd.DataFrame({"compound_id": compound_ids})
    for i in range(n_components):
        col = f"drlatent_{i:02d}"
        out[col] = full_pred[:, i]
        train_mask = out["compound_id"].isin(train_id_set).to_numpy()
        out.loc[train_mask, col] = oof[train_mask, i]

    missing_oof = int(
        np.isnan(out.loc[out["compound_id"].isin(train_id_set)]).any(axis=1).sum()
    )
    if missing_oof:
        raise ValueError(f"Missing OOF latent predictions for {missing_oof} train rows")

    meta = {
        "assay_columns": assay.columns.tolist(),
        "n_components": n_components,
        "min_observed": min_observed,
        "n_estimators": n_estimators,
        "seed": seed,
        "observed_rows": int(full_mask.sum()),
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "folds": fold_stats,
        "full_explained_variance": [
            float(v) for v in full_pipe.named_steps["pca"].explained_variance_ratio_
        ],
    }
    return out, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--min-observed", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = _load_assay_frame()
    print(f"Loaded assay frame: {len(df)} compounds")
    X_chem = _build_chemistry_features(df)
    print(f"Chemistry feature matrix: {X_chem.shape}")
    latent_df, meta = build_latent_predictions(
        df=df,
        X_chem=X_chem,
        n_components=args.n_components,
        min_observed=args.min_observed,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    latent_df.to_parquet(args.out, index=False)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with shape {latent_df.shape}")
    print(f"Wrote {args.meta_out}")


if __name__ == "__main__":
    main()
