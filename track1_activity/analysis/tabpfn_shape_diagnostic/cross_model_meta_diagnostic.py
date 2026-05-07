#!/usr/bin/env python
"""Compare model behavior across shared assay/proximity meta-features."""

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
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "error_anatomy"))
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import umap_split_indices  # noqa: E402

import analyze_tabpfn_shape as diag  # noqa: E402
import error_anatomy as ea  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "cross_model_meta")

MODEL_NAMES = {
    "ensemble": "ens_caruana_bag20",
    "cheme_seed10_top500": (
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
    ),
    "cheme_seed10_default": (
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default"
    ),
    "chemprop_family_meta": "tabpfn_chemprop_family_meta_umap",
    "mordred_singleconc": "lgbm_mordred_singleconc_umap_default",
    "chemprop_embed": "tabpfn_chemprop_pretrain_embed_umap_default",
    "kermt_embed": "tabpfn_kermt_pretrain_embed_umap_default",
    "molformer_c3_embed": "tabpfn_molformer_c3_pretrain_embed_umap",
    "gatedgcn_embed": "tabpfn_gatedgcn_pretrain_embed_umap_default",
    "attentivefp_embed": "tabpfn_attentivefp_pretrain_embed_umap_default",
    "boltz_allpairs": "tabpfn_pooled_boltz_allpairs_umap_default",
    "boltz_core": "tabpfn_pooled_boltz_umap_default",
}

META_FEATURES = [
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
    "has_counter",
    "has_single_conc_hi",
    "has_single_conc_lo",
    "no_aux",
    "near_potent46_t03",
    "near_potent46_t04",
    "counter_above_main",
    "single_hi_low",
    "single_lo_low",
]

SLICE_COLS = [
    "no_aux",
    "has_counter",
    "has_single_conc_hi",
    "has_single_conc_lo",
    "near_potent46_t03",
    "near_potent46_t04",
    "counter_above_main",
    "single_hi_low",
    "single_lo_low",
]


def build_frame() -> pd.DataFrame:
    df, _ = diag.build_model_frame()
    df = diag.add_common_flags(df)
    n = len(df)
    for short, exp_name in MODEL_NAMES.items():
        pred_col = f"pred__{short}"
        if pred_col in df:
            continue
        pred = ea.load_experiment_oof(exp_name, n)
        df[pred_col] = pred
        df[f"residual__{short}"] = df["pec50"].to_numpy(dtype=np.float64) - pred
        df[f"abs_error__{short}"] = np.abs(df[f"residual__{short}"])
    return df


def model_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df["pec50"].to_numpy(dtype=np.float64)
    ens = df["pred__ensemble"].to_numpy(dtype=np.float64)
    for short in MODEL_NAMES:
        pred = df[f"pred__{short}"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": short,
                "experiment": MODEL_NAMES[short],
                "mae": float(np.mean(np.abs(y - pred))),
                "spearman": float(stats.spearmanr(y, pred).statistic),
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
                "corr_vs_ensemble": float(np.corrcoef(pred, ens)[0, 1]),
                "mean_abs_delta_vs_ensemble": float(np.mean(np.abs(pred - ens))),
            }
        )
    return pd.DataFrame(rows).sort_values("mae")


def slice_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slice_col in SLICE_COLS:
        mask = df[slice_col].fillna(False).to_numpy(dtype=bool)
        for short in MODEL_NAMES:
            res = df[f"residual__{short}"].to_numpy(dtype=np.float64)
            true = res[mask]
            false = res[~mask]
            rows.append(
                {
                    "slice": slice_col,
                    "model": short,
                    "n_true": int(mask.sum()),
                    "mae_true": float(np.mean(np.abs(true))) if len(true) else np.nan,
                    "mae_false": float(np.mean(np.abs(false)))
                    if len(false)
                    else np.nan,
                    "mean_residual_true": float(true.mean()) if len(true) else np.nan,
                    "mean_residual_false": float(false.mean())
                    if len(false)
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    ens = out[out["model"].eq("ensemble")][
        ["slice", "mae_true", "mae_false", "mean_residual_true"]
    ].rename(
        columns={
            "mae_true": "ensemble_mae_true",
            "mae_false": "ensemble_mae_false",
            "mean_residual_true": "ensemble_mean_residual_true",
        }
    )
    out = out.merge(ens, on="slice", how="left")
    out["delta_true_mae_vs_ensemble"] = out["mae_true"] - out["ensemble_mae_true"]
    out["delta_false_mae_vs_ensemble"] = out["mae_false"] - out["ensemble_mae_false"]
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[META_FEATURES].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(np.float32)
    return X.astype(np.float32)


def fit_surrogate(
    X: pd.DataFrame, y: np.ndarray, smiles: list[str]
) -> tuple[np.ndarray, lgb.LGBMRegressor]:
    folds = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)
    params = dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    oof = np.zeros(len(X), dtype=np.float64)
    for tr, va in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        oof[va] = model.predict(X.iloc[va])
    full = lgb.LGBMRegressor(**params)
    full.fit(X, y)
    return oof, full


def surrogate_and_shap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = feature_matrix(df)
    rows = []
    shap_rows = []
    background = X.sample(n=min(500, len(X)), random_state=42)
    explain = X.sample(n=min(1000, len(X)), random_state=43)
    for short in MODEL_NAMES:
        y = df[f"pred__{short}"].to_numpy(dtype=np.float64) - df[
            "pred__ensemble"
        ].to_numpy(dtype=np.float64)
        oof, model = fit_surrogate(X, y, df["smiles"].tolist())
        rows.append(
            {
                "model": short,
                "target": "pred_minus_ensemble",
                "target_std": float(np.std(y)),
                "oof_mae": float(mean_absolute_error(y, oof)),
                "oof_r2": float(r2_score(y, oof)),
                "oof_spearman": float(stats.spearmanr(y, oof).statistic),
            }
        )
        explainer = shap.TreeExplainer(model, data=background)
        vals = np.asarray(explainer.shap_values(explain), dtype=np.float64)
        imp = pd.DataFrame(
            {
                "model": short,
                "feature": explain.columns,
                "mean_abs_shap": np.abs(vals).mean(axis=0),
                "mean_shap": vals.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_rows.append(imp)
    return pd.DataFrame(rows), pd.concat(shap_rows, ignore_index=True)


def write_report(
    overall: pd.DataFrame,
    slices: pd.DataFrame,
    surrogate: pd.DataFrame,
    shap_imp: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_DIR.joinpath("overall.csv"), index=False)
    slices.to_csv(OUT_DIR.joinpath("slice_summary.csv"), index=False)
    surrogate.to_csv(OUT_DIR.joinpath("surrogate_metrics.csv"), index=False)
    shap_imp.to_csv(OUT_DIR.joinpath("pred_minus_ensemble_shap.csv"), index=False)

    interesting_slices = []
    for slice_name in ["near_potent46_t04", "near_potent46_t03", "no_aux"]:
        sub = slices[slices["slice"].eq(slice_name)].sort_values("mae_true")
        interesting_slices.append(f"### `{slice_name}`\n")
        interesting_slices.append(
            sub[
                [
                    "model",
                    "n_true",
                    "mae_true",
                    "delta_true_mae_vs_ensemble",
                    "mean_residual_true",
                ]
            ].to_markdown(index=False, floatfmt=".5f")
        )
        interesting_slices.append("")

    lines = [
        "# Cross-Model Meta Diagnostic",
        "",
        "This report does not explain each model's raw input features. It compares",
        "where model OOF predictions differ using shared meta-features: assay labels,",
        "potent46 proximity, and model-disagreement summaries.",
        "",
        "## Overall",
        "",
        overall.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Key Slices",
        "",
        *interesting_slices,
        "## Surrogate Quality For `pred - ensemble`",
        "",
        surrogate.sort_values("oof_r2", ascending=False).to_markdown(
            index=False, floatfmt=".5f"
        ),
        "",
        "## Top SHAP Features Per Model",
        "",
    ]
    for model in overall["model"].tolist():
        sub = shap_imp[shap_imp["model"].eq(model)].head(10)
        lines.extend(
            [
                f"### `{model}`",
                "",
                sub.to_markdown(index=False, floatfmt=".5f"),
                "",
            ]
        )
    OUT_DIR.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_frame()
    overall = model_overall(df)
    slices = slice_summary(df)
    surrogate, shap_imp = surrogate_and_shap(df)
    write_report(overall, slices, surrogate, shap_imp)
    print(f"Wrote cross-model meta diagnostic outputs to {OUT_DIR}")
    print(overall.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
