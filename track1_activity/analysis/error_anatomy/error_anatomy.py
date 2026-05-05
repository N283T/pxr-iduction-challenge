#!/usr/bin/env python
"""Analyze current Track 1 ensemble OOF errors by chemistry and assay slices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

import run_ensemble  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")
ENSEMBLE_NAME = "ens_caruana_bag20"
POTENT_PEC50_THRESHOLD = 6.0
POTENT_SEL_THRESHOLD = 1.5


def mean_abs_error(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.isfinite(arr).sum() == 0:
        return float("nan")
    return float(np.nanmean(np.abs(arr)))


def safe_qcut(values: pd.Series, q: int = 5) -> pd.Series:
    valid = values.dropna()
    if valid.nunique() <= 1:
        return pd.Series(["all"] * len(values), index=values.index, dtype="object")
    try:
        bins = pd.qcut(values, q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index, dtype="object")
    labels = bins.map(lambda x: f"Q{int(x) + 1}" if pd.notna(x) else "missing")
    return labels.astype("object")


def summarize_binary_slice(df: pd.DataFrame, column: str) -> pd.DataFrame:
    mask = df[column].fillna(False).astype(bool)
    true = df.loc[mask]
    false = df.loc[~mask]
    mae_true = mean_abs_error(true["residual"])
    mae_false = mean_abs_error(false["residual"])
    return pd.DataFrame(
        [
            {
                "slice": column,
                "n_true": int(mask.sum()),
                "n_false": int((~mask).sum()),
                "mae_true": mae_true,
                "mae_false": mae_false,
                "delta_mae_true_minus_false": mae_true - mae_false,
                "mean_residual_true": float(true["residual"].mean())
                if len(true)
                else float("nan"),
                "mean_residual_false": float(false["residual"].mean())
                if len(false)
                else float("nan"),
            }
        ]
    )


def tanimoto_max_to_anchors(fps: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    if len(anchor_idx) == 0:
        return np.zeros(len(fps), dtype=np.float64)
    query = fps.astype(bool)
    anchors = query[anchor_idx]
    inter = query.astype(np.uint16) @ anchors.astype(np.uint16).T
    union = query.sum(axis=1)[:, None] + anchors.sum(axis=1)[None, :] - inter
    sim = np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float64),
        where=union > 0,
    )
    return sim.max(axis=1)


def load_experiment_oof(name: str, n_rows: int) -> np.ndarray:
    engine = get_engine()
    exp = pd.read_sql(
        text(
            """
            SELECT id, model_type, hyperparameters
            FROM experiments
            WHERE name = :name
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        engine,
        params={"name": name},
    )
    if exp.empty:
        raise RuntimeError(f"missing experiment: {name}")
    exp_id = int(exp["id"].iloc[0])
    preds = load_oof_predictions(exp_id)
    if preds is None and exp["model_type"].iloc[0] == "ensemble":
        params = exp["hyperparameters"].iloc[0]
        weights = params.get("weights", {}) if isinstance(params, dict) else {}
        if not weights:
            raise RuntimeError(f"ensemble experiment has no weights: {name}")
        matrix = []
        weight_values = []
        for member_name, weight in weights.items():
            if float(weight) == 0.0:
                continue
            matrix.append(load_experiment_oof(member_name, n_rows))
            weight_values.append(float(weight))
        w = np.asarray(weight_values, dtype=np.float64)
        w = w / w.sum()
        preds = np.column_stack(matrix) @ w
    if preds is None:
        raise RuntimeError(f"missing OOF predictions for experiment: {name}")
    if len(preds) != n_rows:
        raise RuntimeError(f"{name}: expected {n_rows} OOF rows, got {len(preds)}")
    return preds.astype(np.float64)


def load_train_metadata() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS train_id,
            t.compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            t.pec50,
            ca.pec50 AS counter_pec50,
            ca.emax_estimate AS counter_emax,
            ca.emax_vs_pos_ctrl AS counter_emax_vs_pos_ctrl,
            sc_hi.log2_fc_estimate AS log2fc_8_25e_6,
            sc_lo.log2_fc_estimate AS log2fc_3_30e_5,
            d.logp,
            d.tpsa,
            d.exactmw,
            d.fractioncsp3,
            d.hba,
            d.hbd,
            d.num_heavy_atoms,
            d.num_heteroatoms,
            d.num_rotatable_bonds,
            d.num_rings,
            d.num_aromatic_rings,
            d.num_bridgehead_atoms,
            d.murcko_scaffold
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
        LEFT JOIN single_concentration sc_hi
               ON sc_hi.compound_id = t.compound_id
              AND sc_hi.concentration_m BETWEEN 8.16e-6 AND 8.34e-6
        LEFT JOIN single_concentration sc_lo
               ON sc_lo.compound_id = t.compound_id
              AND sc_lo.concentration_m BETWEEN 3.27e-5 AND 3.33e-5
        LEFT JOIN compound_descriptors d ON d.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def build_residual_frame() -> tuple[pd.DataFrame, list[str]]:
    df = load_train_metadata()
    n_rows = len(df)
    df["pred"] = load_experiment_oof(ENSEMBLE_NAME, n_rows)
    df["residual"] = df["pec50"] - df["pred"]
    df["abs_error"] = df["residual"].abs()

    member_names = list(run_ensemble.ENSEMBLE_MODELS)
    member_oofs = {name: load_experiment_oof(name, n_rows) for name in member_names}
    member_matrix = np.column_stack([member_oofs[name] for name in member_names])
    df["member_std"] = member_matrix.std(axis=1)
    df["member_range"] = member_matrix.max(axis=1) - member_matrix.min(axis=1)

    family_mask = np.array(
        [
            ("chemprop" in name) or ("cheme_2d_full_boltz_log2fc_pred" in name)
            for name in member_names
        ],
        dtype=bool,
    )
    df["chemprop_family_mean"] = member_matrix[:, family_mask].mean(axis=1)
    df["non_chemprop_mean"] = member_matrix[:, ~family_mask].mean(axis=1)
    df["family_gap"] = df["chemprop_family_mean"] - df["non_chemprop_mean"]
    df["abs_family_gap"] = df["family_gap"].abs()

    counter = df["counter_pec50"]
    selectivity = df["pec50"] - counter
    potent_mask = (df["pec50"] >= POTENT_PEC50_THRESHOLD) & (
        selectivity >= POTENT_SEL_THRESHOLD
    )
    fps = _morgan_fp_matrix(df["smiles"].tolist())
    df["nn_potent46_tanimoto"] = tanimoto_max_to_anchors(
        fps, np.flatnonzero(potent_mask.to_numpy())
    )
    df["is_potent46"] = potent_mask
    return df, member_names


def summarize_quantile_slices(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df:
            continue
        bins = safe_qcut(df[column], q=5)
        for label in sorted(bins.dropna().unique()):
            mask = bins == label
            subset = df.loc[mask]
            rows.append(
                {
                    "variable": column,
                    "bin": label,
                    "n": int(mask.sum()),
                    "value_min": float(subset[column].min()),
                    "value_max": float(subset[column].max()),
                    "mae": mean_abs_error(subset["residual"]),
                    "mean_residual": float(subset["residual"].mean()),
                    "mean_pec50": float(subset["pec50"].mean()),
                    "mean_pred": float(subset["pred"].mean()),
                    "mean_member_std": float(subset["member_std"].mean()),
                }
            )
    return pd.DataFrame(rows)


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["high_member_std_top10"] = out["member_std"] >= out["member_std"].quantile(0.90)
    out["high_family_gap_top10"] = out["abs_family_gap"] >= out[
        "abs_family_gap"
    ].quantile(0.90)
    out["chemprop_family_high_vs_non_top10"] = out["family_gap"] >= out[
        "family_gap"
    ].quantile(0.90)
    out["chemprop_family_low_vs_non_bottom10"] = out["family_gap"] <= out[
        "family_gap"
    ].quantile(0.10)
    out["high_abs_error_top10"] = out["abs_error"] >= out["abs_error"].quantile(0.90)
    out["underpredicted_top10"] = out["residual"] >= out["residual"].quantile(0.90)
    out["overpredicted_bottom10"] = out["residual"] <= out["residual"].quantile(0.10)
    out["has_counter"] = out["counter_pec50"].notna()
    out["counter_above_main"] = (out["counter_pec50"] - out["pec50"]) > 0
    out["has_single_conc_hi"] = out["log2fc_8_25e_6"].notna()
    out["has_single_conc_lo"] = out["log2fc_3_30e_5"].notna()
    out["single_hi_low"] = out["log2fc_8_25e_6"] < 0.3
    out["single_lo_low"] = out["log2fc_3_30e_5"] < 0.3
    out["near_potent46_t04"] = out["nn_potent46_tanimoto"] >= 0.4
    out["near_potent46_t03"] = out["nn_potent46_tanimoto"] >= 0.3
    out["high_logp_top10"] = out["logp"] >= out["logp"].quantile(0.90)
    out["high_mw_top10"] = out["exactmw"] >= out["exactmw"].quantile(0.90)
    out["high_tpsa_top10"] = out["tpsa"] >= out["tpsa"].quantile(0.90)
    return out


def summarize_binary_slices(df: pd.DataFrame) -> pd.DataFrame:
    binary_cols = [
        "high_member_std_top10",
        "high_family_gap_top10",
        "chemprop_family_high_vs_non_top10",
        "chemprop_family_low_vs_non_bottom10",
        "high_abs_error_top10",
        "underpredicted_top10",
        "overpredicted_bottom10",
        "has_counter",
        "counter_above_main",
        "has_single_conc_hi",
        "has_single_conc_lo",
        "single_hi_low",
        "single_lo_low",
        "near_potent46_t04",
        "near_potent46_t03",
        "high_logp_top10",
        "high_mw_top10",
        "high_tpsa_top10",
    ]
    return pd.concat(
        [summarize_binary_slice(df, col) for col in binary_cols],
        ignore_index=True,
    ).sort_values("delta_mae_true_minus_false", ascending=False)


def write_report(
    df: pd.DataFrame,
    quantile: pd.DataFrame,
    binary: pd.DataFrame,
    member_names: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR.joinpath("residuals.csv"), index=False)
    quantile.to_csv(OUT_DIR.joinpath("quantile_slices.csv"), index=False)
    binary.to_csv(OUT_DIR.joinpath("binary_slices.csv"), index=False)
    top_errors = df.sort_values("abs_error", ascending=False).head(80)
    top_errors.to_csv(OUT_DIR.joinpath("top_errors.csv"), index=False)
    gap_cols = [
        "molecule_name",
        "pec50",
        "pred",
        "residual",
        "member_std",
        "family_gap",
        "nn_potent46_tanimoto",
        "logp",
        "exactmw",
        "tpsa",
    ]
    df.sort_values("abs_family_gap", ascending=False).head(80)[gap_cols].to_csv(
        OUT_DIR.joinpath("member_family_gaps.csv"), index=False
    )

    q_spread = (
        quantile.groupby("variable")
        .agg(mae_min=("mae", "min"), mae_max=("mae", "max"), n_bins=("bin", "nunique"))
        .reset_index()
    )
    q_spread["mae_spread"] = q_spread["mae_max"] - q_spread["mae_min"]
    q_spread = q_spread.sort_values("mae_spread", ascending=False)

    report = [
        "# Track 1 Error Anatomy",
        "",
        f"Rows: **{len(df)}**",
        f"Current ensemble: `{ENSEMBLE_NAME}`",
        f"OOF MAE: **{mean_abs_error(df['residual']):.4f}**",
        f"Mean residual (y - pred): **{df['residual'].mean():+.4f}**",
        "",
        "## Current Pool",
        "",
        "\n".join(f"- `{name}`" for name in member_names),
        "",
        "## Binary Slices By MAE Lift",
        "",
        binary.head(18).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Quantile Variables With Largest MAE Spread",
        "",
        q_spread.head(15).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Worst OOF Errors",
        "",
        top_errors[
            [
                "molecule_name",
                "pec50",
                "pred",
                "residual",
                "abs_error",
                "member_std",
                "family_gap",
                "nn_potent46_tanimoto",
                "logp",
                "exactmw",
                "tpsa",
            ]
        ]
        .head(25)
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Initial Read",
        "",
        "- Large positive residual means the current ensemble underpredicts activity.",
        "- Large negative residual means the current ensemble overpredicts activity.",
        "- Prioritize slices with high `delta_mae_true_minus_false` and a plausible internal correction signal.",
    ]
    OUT_DIR.joinpath("report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run() -> None:
    df, member_names = build_residual_frame()
    df = add_binary_flags(df)
    quantile_cols = [
        "pec50",
        "pred",
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
        "log2fc_8_25e_6",
        "log2fc_3_30e_5",
    ]
    quantile = summarize_quantile_slices(df, quantile_cols)
    binary = summarize_binary_slices(df)
    write_report(df, quantile, binary, member_names)
    print(f"Wrote error anatomy outputs to {OUT_DIR}")


if __name__ == "__main__":
    run()
