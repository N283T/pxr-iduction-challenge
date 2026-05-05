#!/usr/bin/env python
"""Cross-fit low-DoF residual models for Track 1 internal correction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_test_smiles  # noqa: E402
from splits import _morgan_fp_matrix, umap_split_indices  # noqa: E402

import error_anatomy as ea  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "residual_model_probe")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
BASE_SUBMISSION = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")
HYBRID_SUBMISSION = SUBMISSION_DIR.joinpath("ens_hybrid_meta_baseline_5050.csv")


def build_lowd_features(df: pd.DataFrame) -> pd.DataFrame:
    pred_centered = df["pred"] - df["pred"].mean()
    no_aux = (
        ~df["has_counter"].astype(bool)
        & ~df["has_single_conc_hi"].astype(bool)
        & ~df["has_single_conc_lo"].astype(bool)
    ).astype(float)
    potent = df["nn_potent46_tanimoto"].astype(float)
    return pd.DataFrame(
        {
            "potent_relu03": np.clip(potent - 0.30, 0, None),
            "potent_relu04": np.clip(potent - 0.40, 0, None),
            "no_aux": no_aux,
            "no_aux_pred_centered": no_aux * pred_centered,
            "member_std": df["member_std"].astype(float),
            "family_gap": df["family_gap"].astype(float),
        },
        index=df.index,
    )


def fit_predict_ridge_residual(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    return model.predict(X_valid).astype(np.float64)


def fit_predict_huber_residual(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000),
    )
    model.fit(X_train, y_train)
    return model.predict(X_valid).astype(np.float64)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "mean_pred": float(np.mean(pred)),
        "mean_residual": float(np.mean(y - pred)),
    }


def build_test_features(
    train_df: pd.DataFrame, base_test_pred: np.ndarray
) -> pd.DataFrame:
    test_df = load_test_smiles()
    potent_idx = np.flatnonzero(train_df["is_potent46"].to_numpy(dtype=bool))
    train_fps = _morgan_fp_matrix(train_df["smiles"].tolist())
    test_fps = _morgan_fp_matrix(test_df["smiles"].tolist())
    test_nn = ea.tanimoto_max_to_anchors(
        np.vstack([train_fps[potent_idx], test_fps]),
        np.arange(len(potent_idx)),
    )[len(potent_idx) :]
    # Blinded test has no counter/single-concentration coverage in local DB.
    no_aux = np.ones(len(test_df), dtype=np.float64)
    pred_centered = base_test_pred - train_df["pred"].mean()
    return pd.DataFrame(
        {
            "potent_relu03": np.clip(test_nn - 0.30, 0, None),
            "potent_relu04": np.clip(test_nn - 0.40, 0, None),
            "no_aux": no_aux,
            "no_aux_pred_centered": no_aux * pred_centered,
            "member_std": np.full(len(test_df), train_df["member_std"].median()),
            "family_gap": np.zeros(len(test_df), dtype=np.float64),
        }
    )


def feature_sets() -> dict[str, list[str]]:
    return {
        "potent_only": ["potent_relu03"],
        "potent_two_knot": ["potent_relu03", "potent_relu04"],
        "potent_noaux": ["potent_relu03", "no_aux", "no_aux_pred_centered"],
        "lowd_core": [
            "potent_relu03",
            "potent_relu04",
            "no_aux",
            "no_aux_pred_centered",
            "member_std",
            "family_gap",
        ],
        "noaux_only": ["no_aux", "no_aux_pred_centered"],
    }


def crossfit_model(
    X: pd.DataFrame,
    residual: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_type: str,
    alpha: float,
) -> np.ndarray:
    out = np.zeros_like(residual)
    for tr, va in folds:
        if model_type == "ridge":
            out[va] = fit_predict_ridge_residual(
                X.iloc[tr], residual[tr], X.iloc[va], alpha
            )
        elif model_type == "huber":
            out[va] = fit_predict_huber_residual(
                X.iloc[tr], residual[tr], X.iloc[va], alpha
            )
        else:
            raise ValueError(f"unknown model_type: {model_type}")
    return out


def fit_full_and_predict_test(
    X: pd.DataFrame,
    residual: np.ndarray,
    X_test: pd.DataFrame,
    model_type: str,
    alpha: float,
) -> np.ndarray:
    if model_type == "ridge":
        return fit_predict_ridge_residual(X, residual, X_test, alpha)
    if model_type == "huber":
        return fit_predict_huber_residual(X, residual, X_test, alpha)
    raise ValueError(f"unknown model_type: {model_type}")


def run_probe() -> pd.DataFrame:
    df, _ = ea.build_residual_frame()
    df = ea.add_binary_flags(df)
    y = df["pec50"].to_numpy(dtype=np.float64)
    raw = df["pred"].to_numpy(dtype=np.float64)
    residual = y - raw
    X_all = build_lowd_features(df)
    raw_metrics = metrics(y, raw)
    folds = umap_split_indices(
        df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    base_sub = pd.read_csv(BASE_SUBMISSION)
    hybrid_sub = pd.read_csv(HYBRID_SUBMISSION)
    base_test = base_sub["pEC50"].to_numpy(dtype=np.float64)
    X_test_all = build_test_features(df, base_test)

    rows = []
    candidate_specs = []
    for set_name, cols in feature_sets().items():
        X = X_all[cols]
        X_test = X_test_all[cols]
        for model_type in ["ridge", "huber"]:
            for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
                shift = crossfit_model(
                    X,
                    residual,
                    folds,
                    model_type=model_type,
                    alpha=alpha,
                )
                for shrink in [0.25, 0.5, 0.75, 1.0]:
                    pred = raw + shrink * shift
                    m = metrics(y, pred)
                    row = {
                        "feature_set": set_name,
                        "model_type": model_type,
                        "alpha": alpha,
                        "shrink": shrink,
                        **m,
                        "delta_mae_vs_raw": m["mae"] - raw_metrics["mae"],
                        "delta_spearman_vs_raw": m["spearman"]
                        - raw_metrics["spearman"],
                        "mean_abs_shift_oof": float(np.mean(np.abs(shrink * shift))),
                        "p90_abs_shift_oof": float(
                            np.quantile(np.abs(shrink * shift), 0.90)
                        ),
                    }
                    rows.append(row)
                    if (
                        row["delta_mae_vs_raw"] <= -0.003
                        and row["delta_spearman_vs_raw"] >= 0.002
                        and row["mean_abs_shift_oof"] <= 0.15
                    ):
                        candidate_specs.append((row, X, X_test, cols))

    result = pd.DataFrame(rows).sort_values(["delta_mae_vs_raw", "model_type"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR.joinpath("summary.csv"), index=False)
    OUT_DIR.joinpath("report.md").write_text(
        "# Residual Model Probe\n\n"
        + result.head(40).to_markdown(index=False, floatfmt=".5f")
        + "\n",
        encoding="utf-8",
    )

    # Materialize a small set of conservative top candidates for inspection.
    materialized = []
    seen = set()
    candidate_specs_sorted = sorted(
        candidate_specs,
        key=lambda spec: (
            spec[0]["delta_mae_vs_raw"],
            spec[0]["mean_abs_shift_oof"],
        ),
    )
    for row, X, X_test, cols in candidate_specs_sorted:
        key = (row["feature_set"], row["model_type"], row["alpha"], row["shrink"])
        if key in seen:
            continue
        seen.add(key)
        if len(materialized) >= 20:
            break
        test_shift = fit_full_and_predict_test(
            X,
            residual,
            X_test,
            model_type=str(row["model_type"]),
            alpha=float(row["alpha"]),
        )
        test_shift *= float(row["shrink"])
        name = (
            "ens_resid_"
            f"{row['feature_set']}_{row['model_type']}"
            f"_a{str(row['alpha']).replace('.', 'p')}"
            f"_s{int(round(float(row['shrink']) * 100)):02d}"
        )
        for anchor_name, anchor_sub in [
            ("raw", base_sub),
            ("hybrid", hybrid_sub),
        ]:
            sub = anchor_sub.copy()
            sub["pEC50"] = sub["pEC50"].to_numpy(dtype=np.float64) + test_shift
            out_path = SUBMISSION_DIR.joinpath(f"{name}_{anchor_name}.csv")
            sub.to_csv(out_path, index=False)
        materialized.append(
            {
                **row,
                "features": ",".join(cols),
                "test_mean_shift": float(np.mean(test_shift)),
                "test_mean_abs_shift": float(np.mean(np.abs(test_shift))),
                "test_p90_abs_shift": float(np.quantile(np.abs(test_shift), 0.90)),
                "test_max_abs_shift": float(np.max(np.abs(test_shift))),
                "raw_submission": f"track1_activity/submissions/{name}_raw.csv",
                "hybrid_submission": f"track1_activity/submissions/{name}_hybrid.csv",
            }
        )
    mat_df = pd.DataFrame(materialized)
    mat_df.to_csv(OUT_DIR.joinpath("materialized_candidates.csv"), index=False)
    if not mat_df.empty:
        OUT_DIR.joinpath("materialized_candidates.md").write_text(
            "# Materialized Residual Candidates\n\n"
            + mat_df.to_markdown(index=False, floatfmt=".5f")
            + "\n",
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    summary = run_probe()
    print(summary.head(40).to_string(index=False))
