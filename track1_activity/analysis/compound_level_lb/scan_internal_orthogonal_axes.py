#!/usr/bin/env python
"""Scan existing internal experiments for lightweight orthogonal axes.

The goal is to find already-materialized, no-external-data model outputs that
are not just another push along the failed id50 direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_train_smiles_target  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402

OUT_DIR = (
    Path(__file__).resolve().parent.joinpath("outputs", "internal_orthogonal_axes")
)
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

EXCLUDE_MARKERS = (
    "admet",
    "drugclip",
    "oe_",
    "openeye",
    "resid_",
    "trial11",
    "no_aux",
    "potent_relu",
)


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Molecule Name" not in df.columns or "pEC50" not in df.columns:
        raise ValueError(f"not a Track 1 submission CSV: {path}")
    return df


def query_experiments() -> pd.DataFrame:
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(
            """
            SELECT e.id, e.name, es.mae_mean::float AS mae,
                   es.spearman_mean::float AS spearman,
                   e.submission_path
            FROM experiments e
            JOIN experiment_summary es ON es.id = e.id
            JOIN experiment_oof_predictions o ON o.experiment_id = e.id
            WHERE e.submission_path IS NOT NULL
              AND e.name NOT LIKE 'ens_%%'
            GROUP BY e.id, e.name, es.mae_mean, es.spearman_mean, e.submission_path
            HAVING count(o.train_idx) = 4140
            ORDER BY es.mae_mean ASC
            """,
            conn,
        )


def is_internal(name: str, path: str) -> bool:
    lower = f"{name} {path}".lower()
    return not any(marker in lower for marker in EXCLUDE_MARKERS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return np.nan
    return float(np.dot(a, b) / denom)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    id48 = load_submission(ID48_PATH)
    id50 = load_submission(ID50_PATH)
    if not (id48["Molecule Name"].to_numpy() == id50["Molecule Name"].to_numpy()).all():
        raise RuntimeError("id48/id50 molecule order mismatch")
    id48_pred = id48["pEC50"].to_numpy(dtype=np.float64)
    id50_direction = id50["pEC50"].to_numpy(dtype=np.float64) - id48_pred
    y = load_train_smiles_target()["pec50"].to_numpy(dtype=np.float64)

    rows = []
    for row in query_experiments().itertuples(index=False):
        if not is_internal(row.name, row.submission_path):
            continue
        path = REPO_ROOT.joinpath(row.submission_path)
        if not path.exists():
            continue
        try:
            sub = load_submission(path)
        except ValueError:
            continue
        if len(sub) != len(id48):
            continue
        if not (
            sub["Molecule Name"].to_numpy() == id48["Molecule Name"].to_numpy()
        ).all():
            continue
        oof = load_oof_predictions(int(row.id))
        if oof is None or len(oof) != len(y):
            continue
        test_pred = sub["pEC50"].to_numpy(dtype=np.float64)
        test_delta = test_pred - id48_pred
        projection = float(
            np.dot(test_delta, id50_direction) / np.dot(id50_direction, id50_direction)
        )
        rows.append(
            {
                "id": int(row.id),
                "name": row.name,
                "submission_path": row.submission_path,
                "oof_mae": float(row.mae),
                "oof_spearman": float(row.spearman),
                "oof_residual_r": float(np.corrcoef(y - oof, y)[0, 1]),
                "test_pearson_vs_id48": float(np.corrcoef(test_pred, id48_pred)[0, 1]),
                "test_spearman_vs_id48": float(
                    stats.spearmanr(test_pred, id48_pred).statistic
                ),
                "mean_abs_shift_vs_id48": float(np.abs(test_delta).mean()),
                "p90_abs_shift_vs_id48": float(np.quantile(np.abs(test_delta), 0.90)),
                "max_abs_shift_vs_id48": float(np.abs(test_delta).max()),
                "projection_on_id50_direction": projection,
                "cosine_vs_id50_direction": cosine(test_delta, id50_direction),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no candidate rows")

    out["orthogonal_score"] = (
        (0.50 - out["oof_mae"]).clip(lower=-0.05, upper=0.10)
        + (1.0 - out["test_pearson_vs_id48"]).clip(lower=0.0, upper=0.10)
        - out["projection_on_id50_direction"].clip(lower=0.0) * 0.01
        - out["mean_abs_shift_vs_id48"].clip(lower=0.0) * 0.02
    )
    out = out.sort_values(
        ["orthogonal_score", "oof_mae", "test_pearson_vs_id48"],
        ascending=[False, True, True],
    )
    out.to_csv(OUT_DIR.joinpath("internal_orthogonal_axis_scan.csv"), index=False)

    filtered = out[
        (out["oof_mae"] <= 0.47)
        & (out["test_pearson_vs_id48"] <= 0.995)
        & (out["mean_abs_shift_vs_id48"] <= 0.20)
        & (out["projection_on_id50_direction"] <= 0.5)
    ].copy()
    filtered.to_csv(OUT_DIR.joinpath("review_candidates.csv"), index=False)

    report = [
        "# Internal Orthogonal Axis Scan",
        "",
        "No external data. This scans existing OOF + test CSV artifacts only.",
        "",
        "## Review Candidates",
        "",
        (
            filtered.head(20)[
                [
                    "name",
                    "oof_mae",
                    "oof_spearman",
                    "test_pearson_vs_id48",
                    "mean_abs_shift_vs_id48",
                    "projection_on_id50_direction",
                    "orthogonal_score",
                    "submission_path",
                ]
            ].to_markdown(index=False, floatfmt=".6f")
            if len(filtered) > 0
            else "No candidate passed the review filter."
        ),
        "",
        "## Top Raw Scores",
        "",
        out.head(25)[
            [
                "name",
                "oof_mae",
                "test_pearson_vs_id48",
                "mean_abs_shift_vs_id48",
                "projection_on_id50_direction",
                "orthogonal_score",
                "submission_path",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
    ]
    OUT_DIR.joinpath("report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote internal orthogonal axis scan to {OUT_DIR}")


if __name__ == "__main__":
    main()
