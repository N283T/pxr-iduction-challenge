#!/usr/bin/env python
"""Build a potent-neighbor plus no-auxiliary correction candidate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_test_smiles  # noqa: E402
from evaluate import compute_metrics, record_experiment, save_oof_predictions  # noqa: E402
from probe_testlike_calibrators import crossfit_affine, fit_affine  # noqa: E402
from splits import _morgan_fp_matrix, umap_split_indices  # noqa: E402

import error_anatomy as ea  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "potent_no_aux_candidate")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
BASE_SUBMISSION = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")


def crossfit_potent_shift(df: pd.DataFrame) -> tuple[np.ndarray, list[float]]:
    y = df["pec50"].to_numpy(dtype=np.float64)
    raw = df["pred"].to_numpy(dtype=np.float64)
    feature = np.clip(
        df["nn_potent46_tanimoto"].to_numpy(dtype=np.float64) - 0.3, 0, None
    )
    folds = umap_split_indices(
        df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    shift = np.zeros_like(raw)
    betas = []
    for tr, va in folds:
        denom = float(np.dot(feature[tr], feature[tr]))
        beta = (
            0.0 if denom == 0.0 else float(np.dot((y - raw)[tr], feature[tr]) / denom)
        )
        betas.append(beta)
        shift[va] = beta * feature[va]
    return shift, betas


def fit_potent_beta(df: pd.DataFrame) -> float:
    y = df["pec50"].to_numpy(dtype=np.float64)
    raw = df["pred"].to_numpy(dtype=np.float64)
    feature = np.clip(
        df["nn_potent46_tanimoto"].to_numpy(dtype=np.float64) - 0.3, 0, None
    )
    denom = float(np.dot(feature, feature))
    return 0.0 if denom == 0.0 else float(np.dot(y - raw, feature) / denom)


def test_potent_feature(train_df: pd.DataFrame, test_smiles: list[str]) -> np.ndarray:
    potent_idx = np.flatnonzero(train_df["is_potent46"].to_numpy(dtype=bool))
    train_fps = _morgan_fp_matrix(train_df["smiles"].tolist())
    test_fps = _morgan_fp_matrix(test_smiles)
    test_nn = ea.tanimoto_max_to_anchors(
        np.vstack([train_fps[potent_idx], test_fps]),
        np.arange(len(potent_idx)),
    )[len(potent_idx) :]
    return np.clip(test_nn - 0.3, 0, None)


def build_candidate(
    alpha_potent: float = 0.5, alpha_no_aux: float = 0.5
) -> pd.DataFrame:
    df, _ = ea.build_residual_frame()
    df = ea.add_binary_flags(df)
    y = df["pec50"].to_numpy(dtype=np.float64)
    raw = df["pred"].to_numpy(dtype=np.float64)
    no_aux = (
        ~df["has_counter"].to_numpy(dtype=bool)
        & ~df["has_single_conc_hi"].to_numpy(dtype=bool)
        & ~df["has_single_conc_lo"].to_numpy(dtype=bool)
    )

    potent_shift, potent_betas = crossfit_potent_shift(df)
    noaux_cal_oof, _ = crossfit_affine(df, no_aux, eval_mask=no_aux)
    noaux_shift = np.zeros_like(raw)
    noaux_shift[no_aux] = noaux_cal_oof[no_aux] - raw[no_aux]
    corrected_oof = raw + alpha_potent * potent_shift + alpha_no_aux * noaux_shift

    potent_beta = fit_potent_beta(df)
    intercept, slope = fit_affine(raw[no_aux], y[no_aux])
    base_sub = pd.read_csv(BASE_SUBMISSION)
    test_df = load_test_smiles()
    base_test = base_sub["pEC50"].to_numpy(dtype=np.float64)
    potent_test_feature = test_potent_feature(df, test_df["smiles"].tolist())
    potent_test_shift = potent_beta * potent_test_feature
    noaux_test_shift = (intercept + slope * base_test) - base_test
    corrected_test = (
        base_test + alpha_potent * potent_test_shift + alpha_no_aux * noaux_test_shift
    )

    name = (
        f"ens_potent_relu03_a{int(round(alpha_potent * 100)):02d}"
        f"_noaux_a{int(round(alpha_no_aux * 100)):02d}"
    )
    out_sub = SUBMISSION_DIR.joinpath(f"{name}.csv")
    sub = base_sub.copy()
    sub["pEC50"] = corrected_test
    sub.to_csv(out_sub, index=False)

    raw_metrics = compute_metrics(y, raw)
    corrected_metrics = compute_metrics(y, corrected_oof)
    summary = pd.DataFrame(
        [
            {
                "name": name,
                "alpha_potent": alpha_potent,
                "alpha_no_aux": alpha_no_aux,
                "raw_mae": raw_metrics["MAE"],
                "corrected_mae": corrected_metrics["MAE"],
                "delta_mae": corrected_metrics["MAE"] - raw_metrics["MAE"],
                "raw_spearman": raw_metrics["Spearman_R"],
                "corrected_spearman": corrected_metrics["Spearman_R"],
                "delta_spearman": corrected_metrics["Spearman_R"]
                - raw_metrics["Spearman_R"],
                "potent_beta_full": potent_beta,
                "potent_beta_cv_mean": float(np.mean(potent_betas)),
                "potent_beta_cv_std": float(np.std(potent_betas)),
                "noaux_intercept": intercept,
                "noaux_slope": slope,
                "test_mean_shift": float(np.mean(corrected_test - base_test)),
                "test_mean_abs_shift": float(
                    np.mean(np.abs(corrected_test - base_test))
                ),
                "test_p90_abs_shift": float(
                    np.quantile(np.abs(corrected_test - base_test), 0.90)
                ),
                "test_max_abs_shift": float(np.max(np.abs(corrected_test - base_test))),
                "test_n_potent_feature_nonzero": int((potent_test_feature > 0).sum()),
                "submission_path": str(out_sub.relative_to(REPO_ROOT)),
            }
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR.joinpath(f"{name}_summary.csv"), index=False)
    detail = pd.DataFrame(
        {
            "molecule_name": df["molecule_name"],
            "pec50": y,
            "raw": raw,
            "corrected_oof": corrected_oof,
            "raw_residual": y - raw,
            "corrected_residual": y - corrected_oof,
            "nn_potent46_tanimoto": df["nn_potent46_tanimoto"],
            "no_aux": no_aux,
            "potent_shift": potent_shift,
            "noaux_shift": noaux_shift,
        }
    )
    detail.to_csv(OUT_DIR.joinpath(f"{name}_oof_detail.csv"), index=False)
    OUT_DIR.joinpath(f"{name}_report.md").write_text(
        "# Potent + No-Aux Candidate\n\n"
        + summary.to_markdown(index=False, floatfmt=".5f")
        + "\n",
        encoding="utf-8",
    )

    exp_id = record_experiment(
        name=name,
        description="Potent-neighbor residual lift plus no-aux affine correction",
        model_type="calibrator",
        feature_set="ens_caruana_bag20_potent_nn_no_aux",
        hyperparameters={
            "alpha_potent": alpha_potent,
            "alpha_no_aux": alpha_no_aux,
            "potent_feature": "max_morgan_tanimoto_to_potent46_minus_0.3_relu",
            "potent_beta": potent_beta,
            "noaux_intercept": intercept,
            "noaux_slope": slope,
            "base_submission": str(BASE_SUBMISSION.relative_to(REPO_ROOT)),
        },
        fold_metrics=[corrected_metrics],
        submission_path=str(out_sub.relative_to(REPO_ROOT)),
        notes=(
            "Cross-fit OOF correction: lift analogs of potent46 and shrink "
            "all no-aux rows. Test has no counter/single-conc coverage."
        ),
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, corrected_oof)
    print(summary.to_string(index=False))
    print(f"Wrote {out_sub}")
    return summary


if __name__ == "__main__":
    build_candidate(alpha_potent=0.5, alpha_no_aux=0.5)
