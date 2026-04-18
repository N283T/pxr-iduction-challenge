#!/usr/bin/env python
"""Who gets Boltz-2 affinity right vs wrong?

Pipeline:
  1. OLS fit: pec50 = a + b * affinity_pred_value on all train
  2. Per-compound residual = pec50 - pred
  3. Split into quartiles by |residual| (Q1 = best, Q4 = worst)
  4. Compare each quartile by molecular descriptors, scaffold frequency,
     and Boltz-2 confidence signals.

Output:
  - reports/boltz2_residual_profile.csv  (quartile summary)
  - reports/boltz2_residual_scaffolds.csv (top scaffolds per quartile)
  - reports/boltz2_residual_profile.png (box plots of descriptor differences)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")
REPORT_DIR = Path(__file__).resolve().parents[1].joinpath("reports")

DESCRIPTOR_COLS = [
    "amw",
    "logp",
    "tpsa",
    "fractioncsp3",
    "hba",
    "hbd",
    "num_heavy_atoms",
    "num_rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
    "num_aliphatic_rings",
    "num_heterocycles",
    "num_spiro_atoms",
    "num_bridgehead_atoms",
    "kappa1",
    "hallkieralpha",
]

BOLTZ_COLS = [
    "iptm",
    "ligand_iptm",
    "complex_plddt",
    "ligand_atom_count",
    "ligand_to_pocket_distance_a",
    "affinity_probability_binary",
    "ensemble_diff",
]


def load_data() -> pd.DataFrame:
    query = """
    SELECT
      t.compound_id,
      t.pec50,
      c.smiles,
      d.murcko_scaffold,
      d.amw, d.logp, d.tpsa, d.fractioncsp3, d.hba, d.hbd,
      d.num_heavy_atoms, d.num_rotatable_bonds, d.num_rings,
      d.num_aromatic_rings, d.num_aliphatic_rings, d.num_heterocycles,
      d.num_spiro_atoms, d.num_bridgehead_atoms, d.kappa1, d.hallkieralpha,
      b.affinity_pred_value,
      b.affinity_pred_value_1,
      b.affinity_pred_value_2,
      b.affinity_probability_binary,
      b.iptm, b.ligand_iptm, b.complex_plddt,
      b.ligand_atom_count, b.ligand_to_pocket_distance_a,
      b.ligand_oversize
    FROM train_activity t
    JOIN compound_boltz2 b ON t.compound_id = b.compound_id
    JOIN compounds c ON t.compound_id = c.id
    LEFT JOIN compound_descriptors d ON t.compound_id = d.compound_id
    WHERE b.affinity_pred_value IS NOT NULL
      AND t.pec50 IS NOT NULL
    ORDER BY t.compound_id
    """
    with psycopg2.connect(**DB_KW) as conn:
        return pd.read_sql(query, conn)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Joined rows: {len(df)}")

    # OLS fit
    x = df["affinity_pred_value"].to_numpy()
    y = df["pec50"].to_numpy()
    b, a = np.polyfit(x, y, deg=1)
    print(f"OLS: pec50 = {a:+.3f} + {b:+.3f} * affinity")
    df["predicted_pec50"] = a + b * df["affinity_pred_value"]
    df["residual"] = df["pec50"] - df["predicted_pec50"]
    df["abs_residual"] = df["residual"].abs()
    df["ensemble_diff"] = (
        df["affinity_pred_value_1"] - df["affinity_pred_value_2"]
    ).abs()

    # Quartile split
    df["quartile"] = pd.qcut(
        df["abs_residual"],
        q=4,
        labels=["Q1_best", "Q2", "Q3", "Q4_worst"],
    )
    # Also signed quartiles (over- vs under-predicted)
    df["residual_sign"] = np.where(df["residual"] >= 0, "under_pred", "over_pred")

    print("\n=== |residual| by quartile ===")
    print(df.groupby("quartile", observed=True)["abs_residual"].describe().round(3))

    # Quartile summary: means of descriptors + Boltz signals
    all_cols = DESCRIPTOR_COLS + BOLTZ_COLS + ["pec50", "affinity_pred_value"]
    summary = (
        df.groupby("quartile", observed=True)[all_cols]
        .mean()
        .round(3)
        .transpose()
    )
    summary["delta_Q4_Q1"] = summary["Q4_worst"] - summary["Q1_best"]
    summary_sorted = summary.reindex(
        summary["delta_Q4_Q1"].abs().sort_values(ascending=False).index
    )
    print("\n=== Mean by quartile (sorted by |Q4 - Q1|) ===")
    print(summary_sorted.to_string())

    summary_sorted.to_csv(REPORT_DIR.joinpath("boltz2_residual_profile.csv"))
    print(f"\nSaved: {REPORT_DIR.joinpath('boltz2_residual_profile.csv')}")

    # Over- vs under-prediction analysis
    print("\n=== Over-pred vs under-pred means ===")
    signed = (
        df.groupby("residual_sign")[all_cols]
        .mean()
        .round(3)
        .transpose()
    )
    signed["delta_under_over"] = signed["under_pred"] - signed["over_pred"]
    signed_sorted = signed.reindex(
        signed["delta_under_over"].abs().sort_values(ascending=False).index
    )
    print(signed_sorted.head(15).to_string())

    # Scaffold frequency in best vs worst quartile
    def top_scaffolds(subset: pd.DataFrame, k: int = 15) -> pd.DataFrame:
        vc = subset["murcko_scaffold"].fillna("(no_scaffold)").value_counts().head(k)
        return vc.rename_axis("murcko_scaffold").reset_index(name="count")

    q1 = df[df["quartile"] == "Q1_best"]
    q4 = df[df["quartile"] == "Q4_worst"]
    print(f"\n=== Top scaffolds in Q1_best (n={len(q1)}) ===")
    print(top_scaffolds(q1).to_string(index=False))
    print(f"\n=== Top scaffolds in Q4_worst (n={len(q4)}) ===")
    print(top_scaffolds(q4).to_string(index=False))

    # Scaffolds that shifted the most (ratio Q4 / Q1, min count 5)
    sc_all = (
        df.groupby("murcko_scaffold", dropna=False)["quartile"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    sc_all["total"] = sc_all[["Q1_best", "Q2", "Q3", "Q4_worst"]].sum(axis=1)
    sc_all = sc_all[sc_all["total"] >= 8]
    sc_all["worst_frac"] = sc_all["Q4_worst"] / sc_all["total"]
    sc_all["best_frac"] = sc_all["Q1_best"] / sc_all["total"]
    worst_tilt = sc_all.sort_values("worst_frac", ascending=False).head(15)
    best_tilt = sc_all.sort_values("best_frac", ascending=False).head(15)
    print("\n=== Scaffolds most concentrated in Q4_worst (total >= 8) ===")
    print(
        worst_tilt[
            ["murcko_scaffold", "total", "Q1_best", "Q4_worst", "worst_frac"]
        ].to_string(index=False)
    )
    print("\n=== Scaffolds most concentrated in Q1_best (total >= 8) ===")
    print(
        best_tilt[
            ["murcko_scaffold", "total", "Q1_best", "Q4_worst", "best_frac"]
        ].to_string(index=False)
    )
    worst_tilt.to_csv(
        REPORT_DIR.joinpath("boltz2_residual_scaffolds_worst.csv"), index=False
    )
    best_tilt.to_csv(
        REPORT_DIR.joinpath("boltz2_residual_scaffolds_best.csv"), index=False
    )

    # Plot: descriptor boxplots by quartile (top 6 by |delta|)
    top6 = summary_sorted.index[:6].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, top6):
        df.boxplot(column=col, by="quartile", ax=ax, showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle(
        f"Descriptor distribution by |residual| quartile (n={len(df)})", fontsize=12
    )
    fig.tight_layout()
    png = REPORT_DIR.joinpath("boltz2_residual_profile.png")
    fig.savefig(png, dpi=140)
    print(f"\nSaved: {png}")

    # Export full per-compound table for further analysis
    out = df[
        [
            "compound_id",
            "pec50",
            "affinity_pred_value",
            "predicted_pec50",
            "residual",
            "abs_residual",
            "quartile",
            "murcko_scaffold",
            "smiles",
        ]
        + DESCRIPTOR_COLS
        + BOLTZ_COLS
    ].copy()
    out.to_csv(
        REPORT_DIR.joinpath("boltz2_residual_per_compound.csv"), index=False
    )
    print(
        f"Saved per-compound table: {REPORT_DIR.joinpath('boltz2_residual_per_compound.csv')}"
    )


if __name__ == "__main__":
    main()
