#!/usr/bin/env python
"""Build a no-auxiliary-assay affine correction candidate submission."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import compute_metrics, record_experiment, save_oof_predictions  # noqa: E402
from probe_testlike_calibrators import crossfit_affine, fit_affine  # noqa: E402

import error_anatomy as ea  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "no_aux_candidate")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
BASE_SUBMISSION = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")


def build_candidate(alpha: float = 0.5) -> pd.DataFrame:
    df, _ = ea.build_residual_frame()
    df = ea.add_binary_flags(df)
    no_aux = (
        ~df["has_counter"].to_numpy(dtype=bool)
        & ~df["has_single_conc_hi"].to_numpy(dtype=bool)
        & ~df["has_single_conc_lo"].to_numpy(dtype=bool)
    )
    y = df["pec50"].to_numpy(dtype=np.float64)
    raw = df["pred"].to_numpy(dtype=np.float64)
    cal_oof, _ = crossfit_affine(df, no_aux, eval_mask=no_aux)
    corrected_oof = raw.copy()
    corrected_oof[no_aux] = (1.0 - alpha) * raw[no_aux] + alpha * cal_oof[no_aux]

    intercept, slope = fit_affine(raw[no_aux], y[no_aux])
    base_sub = pd.read_csv(BASE_SUBMISSION)
    base_test = base_sub["pEC50"].to_numpy(dtype=np.float64)
    corrected_test = (1.0 - alpha) * base_test + alpha * (intercept + slope * base_test)

    name = f"ens_no_aux_affine_a{int(round(alpha * 100)):02d}"
    out_sub = SUBMISSION_DIR.joinpath(f"{name}.csv")
    sub = base_sub.copy()
    sub["pEC50"] = corrected_test
    sub.to_csv(out_sub, index=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = df[
        [
            "molecule_name",
            "pec50",
            "pred",
            "residual",
            "has_counter",
            "has_single_conc_hi",
            "has_single_conc_lo",
        ]
    ].copy()
    detail["no_aux"] = no_aux
    detail["corrected_oof"] = corrected_oof
    detail["corrected_residual"] = y - corrected_oof
    detail.to_csv(OUT_DIR.joinpath(f"{name}_oof_detail.csv"), index=False)

    raw_metrics = compute_metrics(y, raw)
    corrected_metrics = compute_metrics(y, corrected_oof)
    no_aux_raw_metrics = compute_metrics(y[no_aux], raw[no_aux])
    no_aux_corrected_metrics = compute_metrics(y[no_aux], corrected_oof[no_aux])
    summary = pd.DataFrame(
        [
            {
                "scope": "all_train",
                "alpha": alpha,
                "raw_mae": raw_metrics["MAE"],
                "corrected_mae": corrected_metrics["MAE"],
                "delta_mae": corrected_metrics["MAE"] - raw_metrics["MAE"],
                "raw_spearman": raw_metrics["Spearman_R"],
                "corrected_spearman": corrected_metrics["Spearman_R"],
                "delta_spearman": corrected_metrics["Spearman_R"]
                - raw_metrics["Spearman_R"],
                "test_mean_shift": float(np.mean(corrected_test - base_test)),
                "test_max_abs_shift": float(np.max(np.abs(corrected_test - base_test))),
                "intercept": intercept,
                "slope": slope,
            },
            {
                "scope": "no_aux_train",
                "alpha": alpha,
                "raw_mae": no_aux_raw_metrics["MAE"],
                "corrected_mae": no_aux_corrected_metrics["MAE"],
                "delta_mae": no_aux_corrected_metrics["MAE"]
                - no_aux_raw_metrics["MAE"],
                "raw_spearman": no_aux_raw_metrics["Spearman_R"],
                "corrected_spearman": no_aux_corrected_metrics["Spearman_R"],
                "delta_spearman": no_aux_corrected_metrics["Spearman_R"]
                - no_aux_raw_metrics["Spearman_R"],
                "test_mean_shift": float(np.mean(corrected_test - base_test)),
                "test_max_abs_shift": float(np.max(np.abs(corrected_test - base_test))),
                "intercept": intercept,
                "slope": slope,
            },
        ]
    )
    summary.to_csv(OUT_DIR.joinpath(f"{name}_summary.csv"), index=False)
    OUT_DIR.joinpath(f"{name}_report.md").write_text(
        "# No-Aux Affine Candidate\n\n"
        + summary.to_markdown(index=False, floatfmt=".5f")
        + f"\n\nSubmission: `{out_sub.relative_to(REPO_ROOT)}`\n",
        encoding="utf-8",
    )

    exp_id = record_experiment(
        name=name,
        description="No-auxiliary-assay affine correction on current ensemble",
        model_type="calibrator",
        feature_set="ens_caruana_bag20_no_aux_mask",
        hyperparameters={
            "alpha": alpha,
            "fit_mask": "train rows missing counter and both single-concentration assays",
            "intercept": intercept,
            "slope": slope,
            "base_submission": str(BASE_SUBMISSION.relative_to(REPO_ROOT)),
        },
        fold_metrics=[corrected_metrics],
        submission_path=str(out_sub.relative_to(REPO_ROOT)),
        notes=(
            "Test has no counter/single-conc rows, so apply a shrinked affine "
            "fit on no-aux train rows to all test predictions."
        ),
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, corrected_oof)
    print(summary.to_string(index=False))
    print(f"Wrote {out_sub}")
    return summary


if __name__ == "__main__":
    build_candidate(alpha=0.5)
