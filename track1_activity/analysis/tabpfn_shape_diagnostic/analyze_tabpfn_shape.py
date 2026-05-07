#!/usr/bin/env python
"""Analyze high-weight TabPFN errors by assay-shape and proximity slices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "error_anatomy"))
)

from data import get_engine, load_test_smiles  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

import error_anatomy as ea  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")

MODELS = {
    "ensemble": "ens_caruana_bag20",
    "tabpfn_2d_seed10_top500": (
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
    ),
    "tabpfn_2d_seed10_default": (
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default"
    ),
}

COMMON_BINARY_COLS = [
    "has_counter",
    "has_single_conc_hi",
    "has_single_conc_lo",
    "no_aux",
    "counter_above_main",
    "single_hi_low",
    "single_lo_low",
    "near_potent46_t03",
    "near_potent46_t04",
    "high_logp_top10",
    "high_mw_top10",
    "high_tpsa_top10",
    "high_member_std_top10",
    "high_family_gap_top10",
    "chemprop_family_high_vs_non_top10",
    "chemprop_family_low_vs_non_bottom10",
]

QUANTILE_COLS = [
    "pec50",
    "logp",
    "tpsa",
    "exactmw",
    "fractioncsp3",
    "num_heavy_atoms",
    "num_heteroatoms",
    "num_rotatable_bonds",
    "num_rings",
    "member_std",
    "member_range",
    "family_gap",
    "abs_family_gap",
    "nn_potent46_tanimoto",
    "counter_pec50",
    "counter_emax",
    "counter_emax_vs_pos_ctrl",
    "log2fc_8_25e_6",
    "log2fc_3_30e_5",
]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = y - pred
    return {
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "mean_residual": float(np.mean(residual)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


def add_common_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = ea.add_binary_flags(df)
    out["no_aux"] = (
        ~out["has_counter"] & ~out["has_single_conc_hi"] & ~out["has_single_conc_lo"]
    )
    return out


def build_model_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    base, _ = ea.build_residual_frame()
    base = add_common_flags(base)
    y = base["pec50"].to_numpy(dtype=np.float64)
    model_names = {}
    for short_name, experiment_name in MODELS.items():
        pred = ea.load_experiment_oof(experiment_name, len(base))
        base[f"pred__{short_name}"] = pred
        base[f"residual__{short_name}"] = y - pred
        base[f"abs_error__{short_name}"] = np.abs(y - pred)
        model_names[short_name] = experiment_name
    return base, model_names


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df["pec50"].to_numpy(dtype=np.float64)
    for short_name in MODELS:
        pred = df[f"pred__{short_name}"].to_numpy(dtype=np.float64)
        rows.append({"model": short_name, **metrics(y, pred)})
    out = pd.DataFrame(rows)
    ref = out.loc[out["model"].eq("ensemble")].iloc[0]
    out["delta_mae_vs_ensemble"] = out["mae"] - float(ref["mae"])
    out["delta_spearman_vs_ensemble"] = out["spearman"] - float(ref["spearman"])
    return out.sort_values("mae")


def summarize_binary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COMMON_BINARY_COLS:
        if col not in df:
            continue
        mask = df[col].fillna(False).astype(bool)
        for short_name in MODELS:
            res = df[f"residual__{short_name}"]
            true_res = res[mask]
            false_res = res[~mask]
            rows.append(
                {
                    "slice": col,
                    "model": short_name,
                    "n_true": int(mask.sum()),
                    "n_false": int((~mask).sum()),
                    "mae_true": float(np.mean(np.abs(true_res)))
                    if len(true_res)
                    else np.nan,
                    "mae_false": float(np.mean(np.abs(false_res)))
                    if len(false_res)
                    else np.nan,
                    "delta_mae_true_minus_false": (
                        float(np.mean(np.abs(true_res)) - np.mean(np.abs(false_res)))
                        if len(true_res) and len(false_res)
                        else np.nan
                    ),
                    "mean_residual_true": float(true_res.mean())
                    if len(true_res)
                    else np.nan,
                    "mean_residual_false": float(false_res.mean())
                    if len(false_res)
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
    return out.sort_values(["slice", "model"])


def summarize_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in QUANTILE_COLS:
        if col not in df:
            continue
        bins = ea.safe_qcut(df[col], q=5)
        for label in sorted(bins.dropna().unique()):
            mask = bins == label
            subset = df.loc[mask]
            for short_name in MODELS:
                res = subset[f"residual__{short_name}"]
                rows.append(
                    {
                        "variable": col,
                        "bin": label,
                        "model": short_name,
                        "n": int(mask.sum()),
                        "value_min": float(subset[col].min()),
                        "value_max": float(subset[col].max()),
                        "mae": float(np.mean(np.abs(res))),
                        "mean_residual": float(res.mean()),
                        "mean_pred": float(subset[f"pred__{short_name}"].mean()),
                    }
                )
    out = pd.DataFrame(rows)
    ens = out[out["model"].eq("ensemble")][["variable", "bin", "mae"]].rename(
        columns={"mae": "ensemble_mae"}
    )
    out = out.merge(ens, on=["variable", "bin"], how="left")
    out["delta_mae_vs_ensemble"] = out["mae"] - out["ensemble_mae"]
    return out.sort_values(["variable", "bin", "model"])


def load_submission_predictions(experiment_names: dict[str, str]) -> pd.DataFrame:
    engine = get_engine()
    rows = pd.read_sql(
        text(
            """
            SELECT name, submission_path
            FROM experiments
            WHERE name = ANY(:names)
            ORDER BY created_at DESC
            """
        ),
        engine,
        params={"names": list(experiment_names.values())},
    ).drop_duplicates("name", keep="first")
    paths = {row["name"]: row["submission_path"] for _, row in rows.iterrows()}
    test_df = load_test_smiles().copy()
    for short_name, exp_name in experiment_names.items():
        path = paths.get(exp_name)
        if path is None:
            continue
        sub = pd.read_csv(REPO_ROOT.joinpath(path))
        test_df[f"pred__{short_name}"] = sub["pEC50"].to_numpy(dtype=np.float64)
    return test_df


def add_test_proximity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    potent_idx = np.flatnonzero(train_df["is_potent46"].to_numpy(dtype=bool))
    train_fps = _morgan_fp_matrix(train_df["smiles"].tolist())
    test_fps = _morgan_fp_matrix(test_df["smiles"].tolist())
    combined = np.vstack([train_fps[potent_idx], test_fps])
    test_df = test_df.copy()
    test_df["nn_potent46_tanimoto"] = ea.tanimoto_max_to_anchors(
        combined, np.arange(len(potent_idx))
    )[len(potent_idx) :]
    test_df["near_potent46_t03"] = test_df["nn_potent46_tanimoto"] >= 0.3
    test_df["near_potent46_t04"] = test_df["nn_potent46_tanimoto"] >= 0.4
    return test_df


def summarize_test_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for short_name in MODELS:
        pred_col = f"pred__{short_name}"
        if pred_col not in test_df:
            continue
        pred = test_df[pred_col].to_numpy(dtype=np.float64)
        row = {
            "model": short_name,
            "n": len(pred),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std()),
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
        }
        for col in ["near_potent46_t03", "near_potent46_t04"]:
            mask = test_df[col].fillna(False).to_numpy(dtype=bool)
            row[f"{col}_n"] = int(mask.sum())
            row[f"{col}_pred_mean"] = float(pred[mask].mean()) if mask.any() else np.nan
            row[f"{col}_pred_std"] = float(pred[mask].std()) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    overall: pd.DataFrame,
    binary: pd.DataFrame,
    quantile: pd.DataFrame,
    test_summary: pd.DataFrame,
    model_names: dict[str, str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_DIR.joinpath("overall.csv"), index=False)
    binary.to_csv(OUT_DIR.joinpath("binary_slices.csv"), index=False)
    quantile.to_csv(OUT_DIR.joinpath("quantile_slices.csv"), index=False)
    test_summary.to_csv(OUT_DIR.joinpath("test_prediction_summary.csv"), index=False)

    top500_binary = binary[binary["model"].eq("tabpfn_2d_seed10_top500")].copy()
    top500_binary["abs_delta_true_mae_vs_ensemble"] = top500_binary[
        "delta_true_mae_vs_ensemble"
    ].abs()
    top500_worse = top500_binary.sort_values(
        "delta_true_mae_vs_ensemble", ascending=False
    ).head(12)
    top500_better = top500_binary.sort_values("delta_true_mae_vs_ensemble").head(12)

    q_spread = (
        quantile[quantile["model"].eq("tabpfn_2d_seed10_top500")]
        .groupby("variable")
        .agg(mae_min=("mae", "min"), mae_max=("mae", "max"))
        .reset_index()
    )
    q_spread["mae_spread"] = q_spread["mae_max"] - q_spread["mae_min"]
    q_spread = q_spread.sort_values("mae_spread", ascending=False)

    lines = [
        "# TabPFN Shape Diagnostic",
        "",
        "## Models",
        "",
        "\n".join(f"- `{k}`: `{v}`" for k, v in model_names.items()),
        "",
        "## Overall OOF",
        "",
        overall.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Top500 Worse Than Ensemble On True Slice",
        "",
        top500_worse[
            [
                "slice",
                "n_true",
                "mae_true",
                "ensemble_mae_true",
                "delta_true_mae_vs_ensemble",
                "mean_residual_true",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Top500 Better Than Ensemble On True Slice",
        "",
        top500_better[
            [
                "slice",
                "n_true",
                "mae_true",
                "ensemble_mae_true",
                "delta_true_mae_vs_ensemble",
                "mean_residual_true",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Top500 Quantile Variables With Largest MAE Spread",
        "",
        q_spread.head(15).to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Test Prediction Summary",
        "",
        test_summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Read",
        "",
        "- Positive residual means underprediction.",
        "- Negative residual means overprediction.",
        "- Slices where top500 is worse than the ensemble are candidates for a CSV correction against the high-weight 2D axis.",
    ]
    OUT_DIR.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df, model_names = build_model_frame()
    overall = summarize_overall(df)
    binary = summarize_binary(df)
    quantile = summarize_quantiles(df)
    test_df = load_submission_predictions(model_names)
    test_df = add_test_proximity(df, test_df)
    test_summary = summarize_test_predictions(test_df)
    write_report(overall, binary, quantile, test_summary, model_names)
    print(f"Wrote TabPFN shape diagnostic outputs to {OUT_DIR}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
