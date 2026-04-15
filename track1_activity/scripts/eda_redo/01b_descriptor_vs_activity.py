#!/usr/bin/env -S pixi run python
"""Descriptor vs pEC50 for each of the 11 main descriptors.

Pairs with eda_redo_01_descriptor_distributions.py:
  - 01 answers "where is train vs test?"
  - 01b answers "within train, how does activity track each descriptor?"

Each panel is a scatter of the descriptor value vs train_pec50, with
potent compounds (pEC50 >= 6) highlighted and the test min/max bracket
drawn as guide lines. Also reports the Spearman correlation between the
descriptor and pEC50 so we can see which descriptors actually carry
information about activity (vs. drug-likeness biases alone).

Outputs:
  - eda_redo_01b_descriptor_vs_activity.png
  - 01b_descriptor_activity_corr.parquet   - Spearman rho per descriptor
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from eda_redo import POTENT_PEC50, load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

DESCRIPTORS = [
    ("num_heavy_atoms", "Heavy atom count"),
    ("amw", "Molecular weight"),
    ("logp", "logP"),
    ("tpsa", "TPSA"),
    ("num_rotatable_bonds", "Rotatable bonds"),
    ("hba", "HBA"),
    ("hbd", "HBD"),
    ("num_rings", "Rings"),
    ("num_aromatic_rings", "Aromatic rings"),
    ("fractioncsp3", "Fraction sp3"),
    ("num_heteroatoms", "Heteroatoms"),
]


def main() -> None:
    df = load_master()
    tr = df[df["split"] == "train"].copy()
    te = df[df["split"] == "test"].copy()
    print(f"[01b] train N={len(tr):,}  test N={len(te):,}")

    corr_rows = []
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, DESCRIPTORS):
        sub = tr[[col, "train_pec50", "train_is_potent"]].dropna()
        rho = sub[[col, "train_pec50"]].corr(method="spearman").iloc[0, 1]
        pearson = sub[[col, "train_pec50"]].corr(method="pearson").iloc[0, 1]
        corr_rows.append(
            {
                "descriptor": col,
                "spearman_rho": rho,
                "pearson_r": pearson,
                "n": len(sub),
                "test_min": te[col].min(),
                "test_max": te[col].max(),
            }
        )

        pot_mask = sub["train_is_potent"].fillna(False)
        ax.scatter(
            sub.loc[~pot_mask, col],
            sub.loc[~pot_mask, "train_pec50"],
            s=5,
            alpha=0.25,
            color="grey",
        )
        ax.scatter(
            sub.loc[pot_mask, col],
            sub.loc[pot_mask, "train_pec50"],
            s=25,
            alpha=0.9,
            color="C3",
            label=f"potent (N={pot_mask.sum()})",
        )

        t_lo = te[col].min()
        t_hi = te[col].max()
        if pd.notna(t_lo):
            ax.axvline(t_lo, color="C1", linestyle=":", linewidth=1, alpha=0.6)
        if pd.notna(t_hi):
            ax.axvline(t_hi, color="C1", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(POTENT_PEC50, color="C3", linestyle=":", linewidth=1, alpha=0.6)

        ax.set_xlabel(col)
        ax.set_ylabel("train pEC50")
        ax.set_title(f"{label}\nSpearman rho={rho:+.2f}  Pearson r={pearson:+.2f}")
        ax.legend(fontsize=7, loc="lower left")

    # Extra panel: bucketed mean pEC50 per descriptor (see which descriptors are monotone)
    ax = axes[len(DESCRIPTORS)]
    buckets = []
    for col, label in DESCRIPTORS:
        sub = tr[[col, "train_pec50"]].dropna()
        if len(sub) < 20:
            continue
        q = pd.qcut(sub[col], q=5, duplicates="drop")
        grp = sub.groupby(q, observed=True)["train_pec50"].mean()
        buckets.append((label, grp.values))
    for label, vals in buckets:
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=label)
    ax.set_xlabel("descriptor quintile (1=lowest, 5=highest)")
    ax.set_ylabel("mean train pEC50")
    ax.set_title("Mean pEC50 by descriptor quintile")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # Hide any unused axes
    for ax in axes[len(DESCRIPTORS) + 1 :]:
        ax.axis("off")

    fig.suptitle(
        "Issue #52 01b: each descriptor vs pEC50 (potent red, test range in orange)",
        fontsize=13,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR.joinpath("eda_redo_01b_descriptor_vs_activity.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[01b] wrote {fig_path}")

    corr_df = pd.DataFrame(corr_rows).sort_values(
        "spearman_rho", key=abs, ascending=False
    )
    print()
    print("[01b] descriptor vs pEC50 correlations (sorted by |spearman|):")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corr_path = DATA_DIR.joinpath("01b_descriptor_activity_corr.parquet")
    corr_df.to_parquet(corr_path, index=False)
    print(f"[01b] wrote {corr_path}")


if __name__ == "__main__":
    main()
