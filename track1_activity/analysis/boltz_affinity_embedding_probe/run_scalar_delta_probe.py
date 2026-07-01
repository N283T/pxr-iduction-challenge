#!/usr/bin/env -S pixi run python
"""Probe raw Boltz affinity scalar outputs as rank/delta signals.

This is an experiment-only diagnostic. It does not train models, write
submissions, or add rows to the experiment database.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "outputs_scalar_delta"
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine  # noqa: E402


SCALAR_COLS = [
    "affinity_pred_value",
    "affinity_pred_value_1",
    "affinity_pred_value_2",
    "affinity_probability_binary",
    "affinity_probability_binary_1",
    "affinity_probability_binary_2",
    "ensemble_diff_affinity",
    "ensemble_diff_prob",
]


def load_labeled_scalars() -> pd.DataFrame:
    query = """
    WITH labeled AS (
      SELECT t.compound_id, t.pec50, 'train'::text AS split
      FROM train_activity t
      UNION ALL
      SELECT t.compound_id, l.pec50, 'as1'::text AS split
      FROM test_activity_phase1_labels l
      JOIN test_activity t ON t.compound_id = l.compound_id
    )
    SELECT l.split, l.compound_id, l.pec50,
           b.affinity_pred_value,
           b.affinity_pred_value_1,
           b.affinity_pred_value_2,
           b.affinity_probability_binary,
           b.affinity_probability_binary_1,
           b.affinity_probability_binary_2,
           (b.affinity_pred_value_1 - b.affinity_pred_value_2)
               AS ensemble_diff_affinity,
           (b.affinity_probability_binary_1 - b.affinity_probability_binary_2)
               AS ensemble_diff_prob
    FROM labeled l
    JOIN compound_boltz2_affinity_reuse b ON b.compound_id = l.compound_id
    ORDER BY l.split, l.compound_id
    """
    return pd.read_sql(query, get_engine())


def scalar_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in df.groupby("split", sort=True):
        y = group["pec50"].to_numpy(dtype=np.float64)
        for feature in SCALAR_COLS:
            x = group[feature].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "n": len(group),
                    "pearson": stats.pearsonr(x, y).statistic,
                    "spearman": stats.spearmanr(x, y).statistic,
                }
            )
    return pd.DataFrame(rows)


def pairwise_delta_metrics(
    df: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> pd.DataFrame:
    as1 = df[df["split"] == "as1"].copy()
    y = as1["pec50"].to_numpy(dtype=np.float64)
    i, j = np.triu_indices(len(as1), k=1)
    y_delta = y[i] - y[j]

    rows = []
    for feature in SCALAR_COLS:
        x = as1[feature].to_numpy(dtype=np.float64)
        x_delta = x[i] - x[j]
        for threshold in thresholds:
            mask = np.abs(y_delta) >= threshold
            yt = y_delta[mask]
            xt = x_delta[mask]
            auc = roc_auc_score(yt > 0, xt)
            acc = accuracy_score(yt > 0, xt > 0)
            slope, intercept = np.polyfit(xt, yt, 1)
            calibrated = slope * xt + intercept
            rows.append(
                {
                    "feature": feature,
                    "abs_delta_threshold": threshold,
                    "n_pairs": int(mask.sum()),
                    "auc": auc,
                    "auc_best_orientation": max(auc, 1.0 - auc),
                    "sign_accuracy": acc,
                    "sign_accuracy_best_orientation": max(acc, 1.0 - acc),
                    "pearson": stats.pearsonr(xt, yt).statistic,
                    "spearman": stats.spearmanr(xt, yt).statistic,
                    "affine_slope_to_delta_pec50": slope,
                    "raw_delta_mae": np.mean(np.abs(xt - yt)),
                    "affine_delta_mae": np.mean(np.abs(calibrated - yt)),
                }
            )
    return pd.DataFrame(rows)


def write_report(correlations: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(OUT_DIR / "scalar_correlations.csv", index=False)
    pairwise.to_csv(OUT_DIR / "as1_pairwise_delta_metrics.csv", index=False)

    as1_corr = correlations[correlations["split"] == "as1"].sort_values(
        "spearman", ascending=False
    )
    pair_05 = pairwise[pairwise["abs_delta_threshold"] == 0.5].copy()
    pair_05["abs_spearman"] = pair_05["spearman"].abs()
    pair_05 = pair_05.sort_values("abs_spearman", ascending=False).drop(
        columns=["abs_spearman"]
    )

    report = [
        "# Boltz affinity scalar delta probe",
        "",
        "## AS1 scalar correlation with pEC50",
        "",
        as1_corr.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## AS1 pairwise delta metrics, |delta pEC50| >= 0.5",
        "",
        pair_05.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    df = load_labeled_scalars()
    correlations = scalar_correlations(df)
    pairwise = pairwise_delta_metrics(df)
    write_report(correlations, pairwise)
    print(f"Wrote outputs to {OUT_DIR}")
    print(correlations.sort_values(["split", "spearman"]).to_string(index=False))


if __name__ == "__main__":
    main()
