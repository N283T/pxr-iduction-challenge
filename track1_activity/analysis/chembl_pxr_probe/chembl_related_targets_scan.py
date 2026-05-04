#!/usr/bin/env python
"""Cheap ChEMBL related-target scan for external-data triage."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import get_engine  # noqa: E402
from splits import _morgan_fp_matrix, umap_split_indices  # noqa: E402

from chembl_pxr_activation_probe import (  # noqa: E402
    build_nn_features_from_similarity,
    tanimoto_matrix,
)

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "related_targets")
CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"

TARGETS = {
    "PXR_NR1I2": "CHEMBL3401",
    "CAR_NR1I3": "CHEMBL5503",
    "VDR_NR1I1": "CHEMBL1977",
    "FXR_NR1H4": "CHEMBL2047",
    "LXR_family": "CHEMBL3706564",
    "AHR": "CHEMBL3201",
    "GR_NR3C1": "CHEMBL2034",
    "AR_NR3C4": "CHEMBL1871",
    "ER_alpha": "CHEMBL206",
    "PPAR_alpha": "CHEMBL239",
    "PPAR_gamma": "CHEMBL235",
    "RXR_family": "CHEMBL2363070",
}

LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.04,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "n_estimators": 350,
    "verbosity": -1,
    "random_state": 42,
}


def load_challenge() -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_engine()
    train = pd.read_sql(
        """
        SELECT c.molecule_name, c.std_smiles AS smiles, t.pec50, d.inchikey
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        engine,
    )
    test = pd.read_sql(
        """
        SELECT c.molecule_name, c.std_smiles AS smiles, d.inchikey
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        engine,
    )
    return train, test


def load_target_activities(target_id: str) -> pd.DataFrame:
    chembl = create_engine(CHEMBL_URL)
    raw = pd.read_sql(
        """
        SELECT
            s.standard_inchi_key AS inchikey,
            s.canonical_smiles AS smiles,
            m.chembl_id AS mol_chembl_id,
            act.pchembl_value,
            act.standard_type,
            a.assay_type,
            a.confidence_score
        FROM compound_structures s
        JOIN molecule_dictionary m ON m.molregno = s.molregno
        JOIN activities act ON act.molregno = m.molregno
        JOIN assays a ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        WHERE t.chembl_id = %(target_id)s
          AND s.canonical_smiles IS NOT NULL
          AND act.pchembl_value IS NOT NULL
          AND a.confidence_score >= 6
        """,
        chembl,
        params={"target_id": target_id},
    )
    return (
        raw.groupby(["inchikey", "smiles", "mol_chembl_id"], as_index=False)
        .agg(
            chembl_pchembl=("pchembl_value", "median"),
            n_rows=("pchembl_value", "size"),
            assay_types=("assay_type", lambda s: ",".join(sorted(set(s.dropna())))),
            standard_types=(
                "standard_type",
                lambda s: ",".join(sorted(set(s.dropna()))),
            ),
        )
        .reset_index(drop=True)
    )


def build_target_features(
    train: pd.DataFrame, test: pd.DataFrame, external: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext_fp = _morgan_fp_matrix(external["smiles"].tolist())
    values = external["chembl_pchembl"].to_numpy(dtype=np.float32)
    ext_keys = external["inchikey"].to_numpy()

    train_sim = tanimoto_matrix(_morgan_fp_matrix(train["smiles"].tolist()), ext_fp)
    test_sim = tanimoto_matrix(_morgan_fp_matrix(test["smiles"].tolist()), ext_fp)
    train_exact = train["inchikey"].to_numpy()[:, None] == ext_keys[None, :]
    test_exact = test["inchikey"].to_numpy()[:, None] == ext_keys[None, :]
    return (
        build_nn_features_from_similarity(train_sim, values, train_exact),
        build_nn_features_from_similarity(test_sim, values, test_exact),
    )


def score_oof(
    train: pd.DataFrame,
    features: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    feature_cols = list(features.columns)
    X = features[feature_cols].astype(float).to_numpy()
    y = train["pec50"].to_numpy(dtype=np.float64)
    oof = np.zeros_like(y)
    for tr, va in folds:
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
    return {
        "oof_mae": float(mean_absolute_error(y, oof)),
        "oof_spearman": float(stats.spearmanr(y, oof).statistic),
    }


def scan() -> pd.DataFrame:
    train, test = load_challenge()
    folds = umap_split_indices(
        train["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, target_id in TARGETS.items():
        external = load_target_activities(target_id)
        if len(external) < 20:
            rows.append(
                {
                    "target_name": name,
                    "target_id": target_id,
                    "n_external_mols": len(external),
                    "status": "too_few_external_mols",
                }
            )
            continue
        train_features, test_features = build_target_features(train, test, external)
        oof = score_oof(train, train_features, folds)
        row = {
            "target_name": name,
            "target_id": target_id,
            "n_external_mols": len(external),
            "train_exact": int(train_features["chembl_pxr_has_exact_match"].sum()),
            "train_nn_ge_0.4": int(train_features["chembl_pxr_covered_t04"].sum()),
            "test_exact": int(test_features["chembl_pxr_has_exact_match"].sum()),
            "test_nn_ge_0.3": int(test_features["chembl_pxr_covered_t03"].sum()),
            "test_nn_ge_0.4": int(test_features["chembl_pxr_covered_t04"].sum()),
            "test_nn_max": float(test_features["chembl_pxr_nn_tanimoto"].max()),
            "test_nn_median": float(test_features["chembl_pxr_nn_tanimoto"].median()),
            "status": "ok",
            **oof,
        }
        rows.append(row)
        print(
            f"{name} {target_id}: n={len(external)} "
            f"test>=0.4={row['test_nn_ge_0.4']} "
            f"mae={row['oof_mae']:.4f} sp={row['oof_spearman']:.4f}"
        )
    result = pd.DataFrame(rows).sort_values(
        ["oof_mae", "test_nn_ge_0.4"], ascending=[True, False]
    )
    result.to_csv(OUT_DIR.joinpath("related_targets_scan.csv"), index=False)
    OUT_DIR.joinpath("related_targets_scan.md").write_text(
        "# ChEMBL Related Targets Scan\n\n"
        + result.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    df = scan()
    print(df.to_string(index=False))
