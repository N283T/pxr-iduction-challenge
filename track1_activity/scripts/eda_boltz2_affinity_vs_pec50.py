#!/usr/bin/env python
"""Simple diagnostic: Boltz-2 affinity predictions vs experimental pEC50.

Boltz-2 paper (§3, §B.5.2): affinity_value is ~log10(IC50[uM]).
Naive mapping to pEC50 (-log10(EC50[M])): pEC50 ~= 6 - affinity_value.

We report:
  - Pearson/Spearman for main head + 2 ensemble members
  - Same after filtering by ligand_oversize, high-confidence iptm
  - Linear fit (a, b) for pec50 = a + b * affinity_value (OLS)
  - Scatter plot to track1_activity/reports/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import pearsonr, spearmanr

DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")
REPORT_DIR = Path(__file__).resolve().parents[1].joinpath("reports")


def load_joined() -> pd.DataFrame:
    query = """
    SELECT
      t.compound_id,
      t.pec50,
      b.affinity_pred_value,
      b.affinity_pred_value_1,
      b.affinity_pred_value_2,
      b.affinity_probability_binary,
      b.iptm,
      b.ligand_iptm,
      b.complex_plddt,
      b.ligand_oversize,
      b.preprocessing_failed,
      b.ligand_atom_count,
      b.ligand_to_pocket_distance_a
    FROM train_activity t
    JOIN compound_boltz2 b ON t.compound_id = b.compound_id
    WHERE b.affinity_pred_value IS NOT NULL
      AND t.pec50 IS NOT NULL
    ORDER BY t.compound_id
    """
    with psycopg2.connect(**DB_KW) as conn:
        return pd.read_sql(query, conn)


def corr_block(df: pd.DataFrame, label: str) -> dict:
    if len(df) < 10:
        return {"label": label, "n": len(df), "pearson_main": float("nan")}
    r_main, _ = pearsonr(df["pec50"], df["affinity_pred_value"])
    s_main, _ = spearmanr(df["pec50"], df["affinity_pred_value"])
    r_m1, _ = pearsonr(df["pec50"], df["affinity_pred_value_1"])
    r_m2, _ = pearsonr(df["pec50"], df["affinity_pred_value_2"])
    r_mean, _ = pearsonr(
        df["pec50"],
        0.5 * (df["affinity_pred_value_1"] + df["affinity_pred_value_2"]),
    )
    return {
        "label": label,
        "n": len(df),
        "pearson_main": r_main,
        "spearman_main": s_main,
        "pearson_m1": r_m1,
        "pearson_m2": r_m2,
        "pearson_m1m2_mean": r_mean,
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS: y = a + b*x. Returns (a, b, MAE of residuals)."""
    b, a = np.polyfit(x, y, deg=1)
    mae = float(np.mean(np.abs(y - (a + b * x))))
    return float(a), float(b), mae


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    print(f"Joined rows: {len(df)}")
    print(df[["pec50", "affinity_pred_value"]].describe().round(3))

    # Filter variants
    hi_iptm = df[df["iptm"].fillna(0) >= 0.75]
    not_oversize = df[~df["ligand_oversize"].fillna(False)]
    clean = df[
        (df["iptm"].fillna(0) >= 0.75)
        & (~df["ligand_oversize"].fillna(False))
        & (df["ligand_to_pocket_distance_a"].fillna(99) <= 5.0)
    ]

    blocks = [
        corr_block(df, "all_train"),
        corr_block(hi_iptm, "iptm>=0.75"),
        corr_block(not_oversize, "no_oversize"),
        corr_block(clean, "clean(iptm>=0.75+no_oversize+pocket<=5A)"),
    ]
    summary = pd.DataFrame(blocks)
    print("\n=== Correlations ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Linear fit on all_train with naive baseline (pEC50 = 6 - affinity_value)
    x_all = df["affinity_pred_value"].to_numpy()
    y_all = df["pec50"].to_numpy()
    a, b, mae_fit = fit_linear(x_all, y_all)
    mae_naive = float(np.mean(np.abs(y_all - (6.0 - x_all))))
    print("\n=== Scale calibration (all_train) ===")
    print(f"  OLS fit:        pEC50 = {a:+.3f} + {b:+.3f} * affinity   | MAE={mae_fit:.3f}")
    print(f"  Naive baseline: pEC50 = 6.000 + (-1.000) * affinity      | MAE={mae_naive:.3f}")

    # Ensemble diff as uncertainty proxy
    df["ensemble_diff"] = (
        df["affinity_pred_value_1"] - df["affinity_pred_value_2"]
    ).abs()
    # Residual from OLS fit
    df["residual"] = df["pec50"] - (a + b * df["affinity_pred_value"])
    r_unc, _ = pearsonr(df["ensemble_diff"], df["residual"].abs())
    print(
        f"\n|member1 - member2| vs |residual|: Pearson R = {r_unc:.4f} "
        f"(if high -> ensemble diff is an uncertainty proxy)"
    )

    # Binding likelihood (classification) vs pEC50
    r_prob, _ = pearsonr(df["pec50"], df["affinity_probability_binary"])
    print(f"binding_likelihood vs pEC50: Pearson R = {r_prob:.4f}")

    # Save summary
    summary_csv = REPORT_DIR.joinpath("boltz2_affinity_vs_pec50_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    sc = ax.scatter(
        df["affinity_pred_value"],
        df["pec50"],
        c=df["iptm"].fillna(df["iptm"].median()),
        s=8,
        alpha=0.55,
        cmap="viridis",
    )
    plt.colorbar(sc, ax=ax, label="iptm")
    xs = np.linspace(df["affinity_pred_value"].min(), df["affinity_pred_value"].max(), 100)
    ax.plot(xs, a + b * xs, "r-", lw=2, label=f"OLS: {a:.2f} + {b:.2f}x")
    ax.plot(xs, 6.0 - xs, "k--", lw=1.5, label="naive: 6 - x")
    ax.set_xlabel("Boltz-2 affinity_pred_value (~log10 IC50[uM])")
    ax.set_ylabel("experimental pEC50")
    ax.set_title(f"All train (n={len(df)})")
    ax.legend()

    ax2 = axes[1]
    cclean = clean.copy()
    cclean["resid_ols"] = cclean["pec50"] - (a + b * cclean["affinity_pred_value"])
    ax2.scatter(
        cclean["affinity_pred_value"],
        cclean["pec50"],
        c="tab:blue",
        s=8,
        alpha=0.55,
    )
    ax2.plot(xs, a + b * xs, "r-", lw=2)
    ax2.plot(xs, 6.0 - xs, "k--", lw=1.5)
    ax2.set_xlabel("Boltz-2 affinity_pred_value")
    ax2.set_ylabel("experimental pEC50")
    ax2.set_title(f"Clean subset (n={len(cclean)})")

    fig.tight_layout()
    png = REPORT_DIR.joinpath("boltz2_affinity_vs_pec50.png")
    fig.savefig(png, dpi=140)
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
