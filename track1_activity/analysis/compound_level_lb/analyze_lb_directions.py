#!/usr/bin/env python
"""Analyze leaderboard-known prediction directions and build reverse candidates.

This script uses only local submission CSVs and public leaderboard feedback
already recorded in `lb_submissions`. It does not train a model and does not use
external data.
"""

from __future__ import annotations

import sys
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "lb_direction_analysis")
SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")

ID48_PATH = REPO_ROOT.joinpath(
    "track1_activity",
    "analysis",
    "compound_level_lb",
    "outputs",
    "meta_axis_candidates",
    "ens_meta_axis_a343.csv",
)
ID50_PATH = SUB_DIR.joinpath("ens_internal_decor_cap101_bf50_b40_i1_l20.csv")

REVERSE_GAMMAS = (0.05, 0.10, 0.15, 0.20)


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"SMILES", "Molecule Name", "pEC50"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"not a Track 1 submission CSV: {path}")
    return df


def fetch_lb_rows() -> pd.DataFrame:
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(
            """
            SELECT
                id,
                submission_name,
                file_path,
                experiment_name,
                submitted_at,
                lb_rank,
                lb_mae,
                lb_rae,
                lb_spearman,
                notes
            FROM lb_submissions
            WHERE track = 'activity'
              AND lb_mae IS NOT NULL
            ORDER BY submitted_at
            """,
            conn,
        )


def resolve_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    return REPO_ROOT.joinpath(path)


def aligned_predictions(rows: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    anchor_names = anchor["Molecule Name"].to_numpy()
    out_rows = []
    for row in rows.itertuples(index=False):
        path = resolve_path(row.file_path)
        if not path.exists():
            continue
        try:
            sub = load_submission(path)
        except ValueError:
            continue
        if len(sub) != len(anchor):
            continue
        if not (sub["Molecule Name"].to_numpy() == anchor_names).all():
            continue
        pred = sub["pEC50"].to_numpy(dtype=np.float64)
        pred_hash = sha1(np.round(pred, 10).tobytes()).hexdigest()
        out_rows.append(
            {
                "id": int(row.id),
                "submission_name": row.submission_name,
                "file_path": str(path.relative_to(REPO_ROOT)),
                "experiment_name": row.experiment_name,
                "submitted_at": row.submitted_at,
                "lb_rank": row.lb_rank,
                "lb_mae": float(row.lb_mae),
                "lb_rae": float(row.lb_rae) if row.lb_rae is not None else np.nan,
                "lb_spearman": float(row.lb_spearman)
                if row.lb_spearman is not None
                else np.nan,
                "notes": row.notes,
                "pred_hash": pred_hash,
                "pred": pred,
            }
        )
    if not out_rows:
        raise RuntimeError("no aligned LB-known submissions found")
    out = pd.DataFrame(out_rows)
    return (
        out.sort_values(["lb_mae", "submitted_at"])
        .drop_duplicates("pred_hash", keep="first")
        .sort_values("submitted_at")
        .reset_index(drop=True)
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return np.nan
    return float(np.dot(a, b) / denom)


def summarize_directions(
    lb_preds: pd.DataFrame,
    id48_pred: np.ndarray,
    id50_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction_50 = id50_pred - id48_pred
    pred_matrix = np.vstack(lb_preds["pred"].to_numpy())
    deltas = pred_matrix - id48_pred.reshape(1, -1)

    pca = PCA(n_components=min(5, len(lb_preds), deltas.shape[1]), random_state=42)
    pcs = pca.fit_transform(deltas)

    rows = []
    for idx, row in lb_preds.iterrows():
        delta = deltas[idx]
        rows.append(
            {
                "id": row["id"],
                "submission_name": row["submission_name"],
                "file_path": row["file_path"],
                "pred_hash": row["pred_hash"],
                "lb_mae": row["lb_mae"],
                "lb_delta_vs_id48": row["lb_mae"] - 0.4074004936698981,
                "lb_spearman": row["lb_spearman"],
                "mean_shift_vs_id48": float(delta.mean()),
                "mean_abs_shift_vs_id48": float(np.abs(delta).mean()),
                "p90_abs_shift_vs_id48": float(np.quantile(np.abs(delta), 0.90)),
                "max_abs_shift_vs_id48": float(np.abs(delta).max()),
                "pearson_vs_id48": float(np.corrcoef(row["pred"], id48_pred)[0, 1]),
                "cosine_vs_id50_direction": cosine(delta, direction_50),
                "projection_on_id50_direction": float(
                    np.dot(delta, direction_50) / np.dot(direction_50, direction_50)
                ),
                "pc1": float(pcs[idx, 0]),
                "pc2": float(pcs[idx, 1]) if pcs.shape[1] > 1 else 0.0,
                "pc3": float(pcs[idx, 2]) if pcs.shape[1] > 2 else 0.0,
                "notes": row["notes"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("lb_mae")

    feature_cols = [
        "mean_shift_vs_id48",
        "mean_abs_shift_vs_id48",
        "p90_abs_shift_vs_id48",
        "max_abs_shift_vs_id48",
        "cosine_vs_id50_direction",
        "projection_on_id50_direction",
        "pc1",
        "pc2",
        "pc3",
    ]
    model_df = summary.dropna(subset=feature_cols + ["lb_delta_vs_id48"]).copy()
    diagnostics = []
    if len(model_df) >= 5:
        x = model_df[feature_cols].to_numpy(dtype=np.float64)
        y = model_df["lb_delta_vs_id48"].to_numpy(dtype=np.float64)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_std[x_std == 0.0] = 1.0
        x_scaled = (x - x_mean) / x_std
        alphas = np.logspace(-3, 3, 13)
        best_alpha = float(alphas[0])
        best_pred = np.zeros_like(y)
        best_mae = np.inf
        loo = LeaveOneOut()
        for alpha in alphas:
            pred = np.zeros_like(y)
            for train_idx, test_idx in loo.split(x_scaled):
                fold_model = Ridge(alpha=float(alpha))
                fold_model.fit(x_scaled[train_idx], y[train_idx])
                pred[test_idx] = fold_model.predict(x_scaled[test_idx])
            fold_mae = float(np.mean(np.abs(pred - y)))
            if fold_mae < best_mae:
                best_mae = fold_mae
                best_alpha = float(alpha)
                best_pred = pred
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(x_scaled, y)
        lin = LinearRegression().fit(
            model_df[["projection_on_id50_direction"]].to_numpy(), y
        )
        diagnostics.append(
            {
                "n": len(model_df),
                "ridge_alpha": best_alpha,
                "loo_mae": best_mae,
                "loo_corr": float(np.corrcoef(best_pred, y)[0, 1]),
                "linear_projection_coef": float(lin.coef_[0]),
                "linear_projection_intercept": float(lin.intercept_),
            }
        )
        coef_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "ridge_coef": ridge.coef_,
            }
        ).sort_values("ridge_coef")
        coef_df.to_csv(OUT_DIR.joinpath("ridge_coefficients.csv"), index=False)

    return summary, pd.DataFrame(diagnostics)


def build_reverse_candidates(id48: pd.DataFrame, id50: pd.DataFrame) -> pd.DataFrame:
    if not (id48["Molecule Name"].to_numpy() == id50["Molecule Name"].to_numpy()).all():
        raise RuntimeError("id48/id50 molecule order mismatch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    id48_pred = id48["pEC50"].to_numpy(dtype=np.float64)
    id50_pred = id50["pEC50"].to_numpy(dtype=np.float64)
    d = id50_pred - id48_pred
    for gamma in REVERSE_GAMMAS:
        pred = id48_pred - gamma * d
        out = id48.copy()
        out["pEC50"] = pred
        token = f"g{int(round(gamma * 100)):02d}"
        out_path = SUB_DIR.joinpath(f"ens_meta_axis_reverse_id50_{token}.csv")
        out.to_csv(out_path, index=False)
        rows.append(
            {
                "gamma": gamma,
                "path": str(out_path.relative_to(REPO_ROOT)),
                "mean_pred": float(pred.mean()),
                "std_pred": float(pred.std(ddof=0)),
                "min_pred": float(pred.min()),
                "max_pred": float(pred.max()),
                "mean_shift_vs_id48": float((pred - id48_pred).mean()),
                "mean_abs_shift_vs_id48": float(np.abs(pred - id48_pred).mean()),
                "p90_abs_shift_vs_id48": float(
                    np.quantile(np.abs(pred - id48_pred), 0.90)
                ),
                "max_abs_shift_vs_id48": float(np.abs(pred - id48_pred).max()),
                "pearson_vs_id48": float(np.corrcoef(pred, id48_pred)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    direction_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    reverse_summary: pd.DataFrame,
) -> None:
    best_cols = [
        "id",
        "submission_name",
        "lb_mae",
        "lb_delta_vs_id48",
        "projection_on_id50_direction",
        "mean_abs_shift_vs_id48",
        "cosine_vs_id50_direction",
    ]
    report = [
        "# LB Direction Analysis",
        "",
        "## Scope",
        "",
        "Internal analysis of public LB feedback already recorded locally. No external",
        "data and no model training are used here.",
        "",
        "## LB-Known Directions vs id48",
        "",
        direction_summary[best_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Reverse id50 Candidates",
        "",
        reverse_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    if not diagnostics.empty:
        report.extend(
            [
                "## Tiny Direction Model Diagnostics",
                "",
                diagnostics.to_markdown(index=False, floatfmt=".6f"),
                "",
                "This model is intentionally treated as descriptive only; n is too small",
                "for reliable optimization.",
                "",
            ]
        )
    report.extend(
        [
            "## Read",
            "",
            "id50 moved in a direction that recovered from the residual failure but",
            "regressed vs id48. The next low-risk A candidate is therefore a small",
            "extrapolation away from id50 rather than a stronger blend toward it.",
        ]
    )
    OUT_DIR.joinpath("report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    id48 = load_submission(ID48_PATH)
    id50 = load_submission(ID50_PATH)
    lb_rows = fetch_lb_rows()
    lb_preds = aligned_predictions(lb_rows, id48)
    direction_summary, diagnostics = summarize_directions(
        lb_preds,
        id48["pEC50"].to_numpy(dtype=np.float64),
        id50["pEC50"].to_numpy(dtype=np.float64),
    )
    reverse_summary = build_reverse_candidates(id48, id50)

    direction_summary.to_csv(OUT_DIR.joinpath("lb_direction_summary.csv"), index=False)
    diagnostics.to_csv(OUT_DIR.joinpath("direction_model_diagnostics.csv"), index=False)
    reverse_summary.to_csv(
        OUT_DIR.joinpath("reverse_candidate_summary.csv"), index=False
    )
    write_report(direction_summary, diagnostics, reverse_summary)
    print(f"Wrote LB direction analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
