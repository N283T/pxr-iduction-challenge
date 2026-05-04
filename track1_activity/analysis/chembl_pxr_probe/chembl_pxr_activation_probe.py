#!/usr/bin/env python
"""Probe external ChEMBL PXR activation data as cheap Track 1 features."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402
from splits import _morgan_fp_matrix, umap_split_indices  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"

LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "n_estimators": 600,
    "verbosity": -1,
    "random_state": 42,
}


def filter_activation_ec50(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description"].fillna("").str.lower()
    keep_terms = (
        desc.str.contains("activation")
        | desc.str.contains("agonist")
        | desc.str.contains("transactivation")
        | desc.str.contains("cyp3a4")
        | desc.str.contains("luciferase")
    )
    drop_terms = (
        desc.str.contains("antagonist")
        | desc.str.contains("inverse agonist")
        | desc.str.contains("inhibition")
        | desc.str.contains("binding")
        | desc.str.contains("displacement")
    )
    return df[
        df["assay_type"].eq("A")
        & df["standard_type"].eq("EC50")
        & df["confidence_score"].ge(8)
        & df["pchembl_value"].notna()
        & keep_terms
        & ~drop_terms
    ].copy()


def load_chembl_pxr_activities() -> pd.DataFrame:
    chembl = create_engine(CHEMBL_URL)
    raw = pd.read_sql(
        """
        SELECT
            s.standard_inchi_key AS inchikey,
            s.canonical_smiles AS smiles,
            m.chembl_id AS mol_chembl_id,
            a.chembl_id AS assay_chembl_id,
            a.assay_type,
            a.confidence_score,
            a.description,
            act.standard_type,
            act.standard_relation,
            act.standard_value,
            act.standard_units,
            act.pchembl_value
        FROM compound_structures s
        JOIN molecule_dictionary m ON m.molregno = s.molregno
        JOIN activities act ON act.molregno = m.molregno
        JOIN assays a ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        WHERE t.chembl_id = 'CHEMBL3401'
          AND s.canonical_smiles IS NOT NULL
          AND act.pchembl_value IS NOT NULL
        """,
        chembl,
    )
    filtered = filter_activation_ec50(raw)
    # Collapse duplicate molecule rows after filtering. Median is robust to
    # duplicated assay rows and keeps one external value per ChEMBL molecule.
    return (
        filtered.groupby(["inchikey", "smiles", "mol_chembl_id"], as_index=False)
        .agg(
            chembl_pxr_pchembl=("pchembl_value", "median"),
            n_chembl_pxr_rows=("pchembl_value", "size"),
            assays=("assay_chembl_id", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index(drop=True)
    )


def tanimoto_matrix(
    query: np.ndarray, ref: np.ndarray, chunk_size: int = 512
) -> np.ndarray:
    query_bool = query.astype(bool)
    ref_bool = ref.astype(bool)
    ref_sum = ref_bool.sum(axis=1)[None, :]
    chunks = []
    for start in range(0, len(query_bool), chunk_size):
        q = query_bool[start : start + chunk_size]
        inter = q.astype(np.uint16) @ ref_bool.astype(np.uint16).T
        union = q.sum(axis=1)[:, None] + ref_sum - inter
        sim = np.divide(
            inter,
            union,
            out=np.zeros_like(inter, dtype=np.float32),
            where=union > 0,
        )
        chunks.append(sim)
    return np.vstack(chunks)


def topk_weighted_values(sim: np.ndarray, values: np.ndarray, k: int = 5) -> np.ndarray:
    if sim.shape[1] == 0:
        return np.full(sim.shape[0], np.nan)
    k_eff = min(k, sim.shape[1])
    idx = np.argpartition(sim, kth=sim.shape[1] - k_eff, axis=1)[:, -k_eff:]
    row = np.arange(sim.shape[0])[:, None]
    top_sim = sim[row, idx].astype(np.float64)
    top_values = values[idx].astype(np.float64)
    denom = top_sim.sum(axis=1)
    weighted = (top_sim * top_values).sum(axis=1)
    return np.divide(
        weighted, denom, out=np.full(sim.shape[0], np.nan), where=denom > 0
    )


def build_nn_features_from_similarity(
    sim: np.ndarray, values: np.ndarray, exact_match: np.ndarray, k: int = 5
) -> pd.DataFrame:
    working = sim.copy()
    has_exact = exact_match.any(axis=1)
    working[exact_match] = -np.inf
    nn_idx = np.argmax(working, axis=1)
    nn_sim = working[np.arange(len(working)), nn_idx]
    nn_sim = np.where(np.isfinite(nn_sim), nn_sim, 0.0)
    nn_pchembl = values[nn_idx]
    topk = topk_weighted_values(np.maximum(working, 0.0), values, k=k)
    return pd.DataFrame(
        {
            "chembl_pxr_nn_tanimoto": nn_sim,
            "chembl_pxr_nn_pchembl": nn_pchembl,
            "chembl_pxr_top5_pchembl": topk,
            "chembl_pxr_has_exact_match": has_exact,
            "chembl_pxr_covered_t03": nn_sim >= 0.30,
            "chembl_pxr_covered_t04": nn_sim >= 0.40,
        }
    )


def build_feature_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chembl = load_chembl_pxr_activities()
    pxr_engine = get_engine()
    train = pd.read_sql(
        """
        SELECT c.molecule_name, c.std_smiles AS smiles, t.pec50, d.inchikey
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        pxr_engine,
    )
    test = pd.read_sql(
        """
        SELECT c.molecule_name, c.std_smiles AS smiles, d.inchikey
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        pxr_engine,
    )

    chem_fp = _morgan_fp_matrix(chembl["smiles"].tolist())
    values = chembl["chembl_pxr_pchembl"].to_numpy(dtype=np.float32)
    chem_keys = chembl["inchikey"].to_numpy()

    train_sim = tanimoto_matrix(_morgan_fp_matrix(train["smiles"].tolist()), chem_fp)
    test_sim = tanimoto_matrix(_morgan_fp_matrix(test["smiles"].tolist()), chem_fp)
    train_exact = train["inchikey"].to_numpy()[:, None] == chem_keys[None, :]
    test_exact = test["inchikey"].to_numpy()[:, None] == chem_keys[None, :]

    train_features = pd.concat(
        [
            train[["molecule_name", "smiles", "pec50", "inchikey"]].reset_index(
                drop=True
            ),
            build_nn_features_from_similarity(
                train_sim, values, train_exact
            ).reset_index(drop=True),
        ],
        axis=1,
    )
    test_features = pd.concat(
        [
            test[["molecule_name", "smiles", "inchikey"]].reset_index(drop=True),
            build_nn_features_from_similarity(test_sim, values, test_exact).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    return chembl, train_features, test_features


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "spearman": float(stats.spearmanr(y_true, pred).statistic),
    }


def run_oof_models(train_features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "chembl_pxr_nn_tanimoto",
        "chembl_pxr_nn_pchembl",
        "chembl_pxr_top5_pchembl",
        "chembl_pxr_has_exact_match",
        "chembl_pxr_covered_t03",
        "chembl_pxr_covered_t04",
    ]
    X = train_features[feature_cols].astype(float).to_numpy()
    y = train_features["pec50"].to_numpy(dtype=np.float64)
    folds = umap_split_indices(
        train_features["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )

    oof_lgbm = np.zeros_like(y)
    oof_ridge = np.zeros_like(y)
    for fold, (tr, va) in enumerate(folds):
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X[tr], y[tr])
        oof_lgbm[va] = model.predict(X[va])

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X[tr], y[tr])
        oof_ridge[va] = ridge.predict(X[va])
        print(
            f"fold {fold}: lgbm_mae={mean_absolute_error(y[va], oof_lgbm[va]):.4f} "
            f"ridge_mae={mean_absolute_error(y[va], oof_ridge[va]):.4f}"
        )

    rows = []
    for name, pred in [("chembl_pxr_lgbm", oof_lgbm), ("chembl_pxr_ridge", oof_ridge)]:
        row = {"model": name, **metrics(y, pred)}
        rows.append(row)

    # Residual probe against latest stored calibrated ensemble OOF if present.
    base_id = pd.read_sql(
        """
        SELECT e.id
        FROM experiments e
        JOIN experiment_oof_predictions o ON o.experiment_id = e.id
        WHERE e.name IN ('ens_caruana_bag20_calibrated_best',
                         'ens_caruana_bag20_calibrated_linear_pos')
        GROUP BY e.id
        HAVING count(o.train_idx) = 4140
        ORDER BY e.id DESC
        LIMIT 1
        """,
        get_engine(),
    )
    if not base_id.empty:
        base = load_oof_predictions(int(base_id.iloc[0]["id"]))
        base_metrics = metrics(y, base)
        rows.append({"model": "reference_ensemble_oof", **base_metrics})
        for name, pred in [("lgbm", oof_lgbm), ("ridge", oof_ridge)]:
            residual_r = float(np.corrcoef(y - base, y - pred)[0, 1])
            for alpha in [0.01, 0.02, 0.05, 0.10, 0.20]:
                blend = (1 - alpha) * base + alpha * pred
                row = {
                    "model": f"reference_plus_{alpha:.2f}_{name}",
                    **metrics(y, blend),
                    "delta_mae_vs_reference": float(
                        mean_absolute_error(y, blend) - base_metrics["mae"]
                    ),
                    "residual_r_vs_reference": residual_r,
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "oof_model_summary.csv", index=False)
    train_features.assign(oof_lgbm=oof_lgbm, oof_ridge=oof_ridge).to_csv(
        OUT_DIR / "train_features_with_oof.csv", index=False
    )
    return out


def write_report(
    chembl: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    oof_summary: pd.DataFrame,
) -> None:
    coverage = pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(train_features),
                "exact": int(train_features["chembl_pxr_has_exact_match"].sum()),
                "nn_ge_0.3": int(train_features["chembl_pxr_covered_t03"].sum()),
                "nn_ge_0.4": int(train_features["chembl_pxr_covered_t04"].sum()),
                "nn_max": float(train_features["chembl_pxr_nn_tanimoto"].max()),
                "nn_median": float(train_features["chembl_pxr_nn_tanimoto"].median()),
            },
            {
                "split": "test",
                "n": len(test_features),
                "exact": int(test_features["chembl_pxr_has_exact_match"].sum()),
                "nn_ge_0.3": int(test_features["chembl_pxr_covered_t03"].sum()),
                "nn_ge_0.4": int(test_features["chembl_pxr_covered_t04"].sum()),
                "nn_max": float(test_features["chembl_pxr_nn_tanimoto"].max()),
                "nn_median": float(test_features["chembl_pxr_nn_tanimoto"].median()),
            },
        ]
    )
    coverage.to_csv(OUT_DIR / "coverage_summary.csv", index=False)
    report = [
        "# ChEMBL PXR Activation Probe",
        "",
        f"Filtered ChEMBL activation EC50 molecules: **{len(chembl)}**",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOF Models",
        "",
        oof_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "This is a cheap external-data feature probe only. No submission CSV is produced.",
        "The primary pass criterion is a strong OOF gain after blending with the current",
        "ensemble reference; weak standalone or positive delta MAE should be closed.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chembl, train_features, test_features = build_feature_matrices()
    chembl.to_csv(OUT_DIR / "chembl_pxr_activation_ec50.csv", index=False)
    train_features.to_csv(OUT_DIR / "train_chembl_pxr_features.csv", index=False)
    test_features.to_csv(OUT_DIR / "test_chembl_pxr_features.csv", index=False)
    print(f"Filtered ChEMBL activation EC50 molecules: {len(chembl)}")
    print(
        "Coverage test: "
        f"exact={int(test_features['chembl_pxr_has_exact_match'].sum())} "
        f"nn>=0.4={int(test_features['chembl_pxr_covered_t04'].sum())}"
    )
    summary = run_oof_models(train_features)
    write_report(chembl, train_features, test_features, summary)
    print(f"Wrote ChEMBL PXR probe outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
