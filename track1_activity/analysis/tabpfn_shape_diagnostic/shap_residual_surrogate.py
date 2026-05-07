#!/usr/bin/env python
"""Tree SHAP diagnostics for TabPFN residual shape/proximity surrogates."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "error_anatomy"))
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import umap_split_indices  # noqa: E402

import analyze_tabpfn_shape as diag  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "shap_residual_surrogate")

FEATURE_COLS = [
    "pec50",
    "logp",
    "tpsa",
    "exactmw",
    "fractioncsp3",
    "hba",
    "hbd",
    "num_heavy_atoms",
    "num_heteroatoms",
    "num_rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
    "num_bridgehead_atoms",
    "counter_pec50",
    "counter_emax",
    "counter_emax_vs_pos_ctrl",
    "log2fc_8_25e_6",
    "log2fc_3_30e_5",
    "member_std",
    "member_range",
    "family_gap",
    "abs_family_gap",
    "nn_potent46_tanimoto",
    "is_potent46",
    "has_counter",
    "has_single_conc_hi",
    "has_single_conc_lo",
    "no_aux",
    "counter_above_main",
    "single_hi_low",
    "single_lo_low",
    "near_potent46_t03",
    "near_potent46_t04",
]

TARGETS = {
    "top500_residual": "residual__tabpfn_2d_seed10_top500",
    "top500_abs_error_minus_ensemble": None,
    "top500_pred_minus_ensemble": None,
}


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(np.float32)
    X = X.astype(np.float32)
    # LightGBM and Tree SHAP both handle NaN; keep missingness informative.
    return X


def target_values(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "top500_abs_error_minus_ensemble":
        return df["abs_error__tabpfn_2d_seed10_top500"].to_numpy(dtype=np.float64) - df[
            "abs_error__ensemble"
        ].to_numpy(dtype=np.float64)
    if name == "top500_pred_minus_ensemble":
        return df["pred__tabpfn_2d_seed10_top500"].to_numpy(dtype=np.float64) - df[
            "pred__ensemble"
        ].to_numpy(dtype=np.float64)
    col = TARGETS[name]
    if col is None:
        raise ValueError(name)
    return df[col].to_numpy(dtype=np.float64)


def fit_oof_surrogate(
    X: pd.DataFrame, y: np.ndarray, smiles: list[str]
) -> tuple[np.ndarray, list[lgb.LGBMRegressor], pd.DataFrame]:
    folds = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)
    oof = np.zeros(len(X), dtype=np.float64)
    models = []
    rows = []
    params = dict(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    for fold, (tr, va) in enumerate(folds):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(X.iloc[va])
        oof[va] = pred
        models.append(model)
        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_valid": len(va),
                "mae": float(mean_absolute_error(y[va], pred)),
                "r2": float(r2_score(y[va], pred)),
                "spearman": float(stats.spearmanr(y[va], pred).statistic),
                "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            }
        )
    return oof, models, pd.DataFrame(rows)


def shap_summary(
    model: lgb.LGBMRegressor, X: pd.DataFrame, target_name: str
) -> pd.DataFrame:
    background = X.sample(n=min(500, len(X)), random_state=42)
    explain = X.sample(n=min(1000, len(X)), random_state=43)
    explainer = shap.TreeExplainer(model, data=background)
    values = explainer.shap_values(explain)
    arr = np.asarray(values, dtype=np.float64)
    out = pd.DataFrame(
        {
            "target": target_name,
            "feature": explain.columns,
            "mean_abs_shap": np.abs(arr).mean(axis=0),
            "mean_shap": arr.mean(axis=0),
            "feature_mean": explain.mean(axis=0).to_numpy(dtype=np.float64),
            "feature_missing_frac": explain.isna()
            .mean(axis=0)
            .to_numpy(dtype=np.float64),
        }
    )
    return out.sort_values("mean_abs_shap", ascending=False)


def train_full_model(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=4242,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def write_report(
    cv_rows: pd.DataFrame, shap_rows: pd.DataFrame, target_metrics: pd.DataFrame
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_rows.to_csv(OUT_DIR.joinpath("cv_metrics.csv"), index=False)
    shap_rows.to_csv(OUT_DIR.joinpath("shap_feature_importance.csv"), index=False)
    target_metrics.to_csv(OUT_DIR.joinpath("target_metrics.csv"), index=False)

    lines = [
        "# SHAP Residual Surrogate",
        "",
        "LightGBM surrogates are trained on low-dimensional assay-shape, proximity,",
        "chemistry, and ensemble-disagreement features, then explained with Tree SHAP.",
        "",
        "## Surrogate OOF Metrics",
        "",
        target_metrics.to_markdown(index=False, floatfmt=".5f"),
        "",
    ]
    for target in TARGETS:
        sub = shap_rows[shap_rows["target"].eq(target)].head(20)
        lines.extend(
            [
                f"## Top SHAP Features: `{target}`",
                "",
                sub.to_markdown(index=False, floatfmt=".5f"),
                "",
            ]
        )
    OUT_DIR.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, _ = diag.build_model_frame()
    X = build_feature_matrix(df)
    all_cv = []
    all_shap = []
    metrics_rows = []
    for target_name in TARGETS:
        y = target_values(df, target_name)
        oof, _, cv = fit_oof_surrogate(X, y, df["smiles"].tolist())
        cv.insert(0, "target", target_name)
        all_cv.append(cv)
        metrics_rows.append(
            {
                "target": target_name,
                "target_std": float(np.std(y)),
                "oof_mae": float(mean_absolute_error(y, oof)),
                "oof_r2": float(r2_score(y, oof)),
                "oof_spearman": float(stats.spearmanr(y, oof).statistic),
            }
        )
        full_model = train_full_model(X, y)
        all_shap.append(shap_summary(full_model, X, target_name))
    cv_rows = pd.concat(all_cv, ignore_index=True)
    shap_rows = pd.concat(all_shap, ignore_index=True)
    target_metrics = pd.DataFrame(metrics_rows)
    write_report(cv_rows, shap_rows, target_metrics)
    print(f"Wrote SHAP residual surrogate outputs to {OUT_DIR}")
    print(target_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
