#!/usr/bin/env python
"""Probe low-DoF calibrators fitted on train rows that resemble blinded test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import umap_split_indices  # noqa: E402

import error_anatomy as ea  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "testlike_calibrators")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "mean_pred": float(np.mean(pred)),
        "mean_residual": float(np.mean(y - pred)),
    }


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return float(model.intercept_), float(model.coef_[0])


def crossfit_affine(
    df: pd.DataFrame, fit_mask: np.ndarray, eval_mask: np.ndarray | None = None
) -> tuple[np.ndarray, list[dict[str, float]]]:
    y = df["pec50"].to_numpy(dtype=np.float64)
    pred = df["pred"].to_numpy(dtype=np.float64)
    folds = umap_split_indices(
        df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    out = np.zeros_like(pred)
    params = []
    for fold, (tr, va) in enumerate(folds):
        tr_fit = tr[fit_mask[tr]]
        if len(tr_fit) < 50:
            raise RuntimeError(f"fold {fold}: fit mask too small ({len(tr_fit)})")
        intercept, slope = fit_affine(pred[tr_fit], y[tr_fit])
        out[va] = intercept + slope * pred[va]
        va_eval = va if eval_mask is None else va[eval_mask[va]]
        fold_metrics = metrics(y[va_eval], out[va_eval])
        params.append(
            {
                "fold": fold,
                "n_fit": int(len(tr_fit)),
                "n_eval": int(len(va_eval)),
                "intercept": intercept,
                "slope": slope,
                **fold_metrics,
            }
        )
    return out, params


def probe() -> pd.DataFrame:
    df, _ = ea.build_residual_frame()
    df = ea.add_binary_flags(df)
    y = df["pec50"].to_numpy(dtype=np.float64)
    base = df["pred"].to_numpy(dtype=np.float64)

    masks = {
        "all_train": np.ones(len(df), dtype=bool),
        "no_counter": ~df["has_counter"].to_numpy(dtype=bool),
        "no_single_hi": ~df["has_single_conc_hi"].to_numpy(dtype=bool),
        "no_single_lo": ~df["has_single_conc_lo"].to_numpy(dtype=bool),
        "no_aux_all": (
            ~df["has_counter"].to_numpy(dtype=bool)
            & ~df["has_single_conc_hi"].to_numpy(dtype=bool)
            & ~df["has_single_conc_lo"].to_numpy(dtype=bool)
        ),
        "near_potent46_t04": df["near_potent46_t04"].to_numpy(dtype=bool),
        "member_std_top10": df["high_member_std_top10"].to_numpy(dtype=bool),
        "family_gap_top10": df["high_family_gap_top10"].to_numpy(dtype=bool),
    }

    rows = [{"candidate": "raw", "fit_mask": "none", **metrics(y, base)}]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        if int(mask.sum()) < 80:
            continue
        calibrated, fold_params = crossfit_affine(df, mask)
        row = {
            "candidate": f"affine_fit_{name}_eval_all",
            "fit_mask": name,
            "n_fit_total": int(mask.sum()),
            **metrics(y, calibrated),
        }
        row["delta_mae_vs_raw"] = row["mae"] - rows[0]["mae"]
        row["delta_spearman_vs_raw"] = row["spearman"] - rows[0]["spearman"]
        rows.append(row)

        calibrated_eval, fold_params_eval = crossfit_affine(df, mask, eval_mask=mask)
        eval_metrics = metrics(y[mask], calibrated_eval[mask])
        raw_eval_metrics = metrics(y[mask], base[mask])
        rows.append(
            {
                "candidate": f"affine_fit_{name}_eval_same_mask",
                "fit_mask": name,
                "n_fit_total": int(mask.sum()),
                **eval_metrics,
                "delta_mae_vs_raw": eval_metrics["mae"] - raw_eval_metrics["mae"],
                "delta_spearman_vs_raw": eval_metrics["spearman"]
                - raw_eval_metrics["spearman"],
            }
        )
        pd.DataFrame(fold_params).to_csv(
            OUT_DIR.joinpath(f"{name}_fold_params_eval_all.csv"), index=False
        )
        pd.DataFrame(fold_params_eval).to_csv(
            OUT_DIR.joinpath(f"{name}_fold_params_eval_same_mask.csv"), index=False
        )

    result = pd.DataFrame(rows).sort_values("mae")
    result.to_csv(OUT_DIR.joinpath("summary.csv"), index=False)
    OUT_DIR.joinpath("report.md").write_text(
        "# Test-Like Calibrator Probe\n\n"
        + result.to_markdown(index=False, floatfmt=".5f")
        + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    summary = probe()
    print(summary.to_string(index=False))
