#!/usr/bin/env -S pixi run python
"""Size distribution comparison of train vs test compounds.

Outputs (all under docs/figures/ and data/eda_redo/):
  - eda_redo_01_size_dist.png       - 4-panel: HA/MW/Rings distributions + HA-pEC50 scatter
  - 01_size_flagged.parquet         - train compounds larger than any test compound,
                                       joined with their pEC50 so we do not drop
                                       genuine potent hits just because they are big.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from eda_redo import POTENT_PEC50, load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")


def main() -> None:
    df = load_master()
    tr = df[df["split"] == "train"].copy()
    te = df[df["split"] == "test"].copy()

    # ------------------------------------------------------------------
    # 1. Distribution tables (train vs test)
    # ------------------------------------------------------------------
    print(f"[01] train N = {len(tr):,}  |  test N = {len(te):,}")
    for col in ("num_heavy_atoms", "amw", "num_rings", "logp"):
        print(
            f"  {col:<20s}  train {tr[col].min():6.1f}..{tr[col].max():6.1f} (median {tr[col].median():6.1f})"
        )
        print(
            f"  {col:<20s}  test  {te[col].min():6.1f}..{te[col].max():6.1f} (median {te[col].median():6.1f})"
        )

    # Largest train vs max-test thresholds are the critical cutoffs.
    ha_test_max = te["num_heavy_atoms"].max()
    mw_test_max = te["amw"].max()
    print()
    print(f"[01] test max HA = {ha_test_max:.0f}  |  test max MW = {mw_test_max:.1f}")
    oversize = tr[
        (tr["num_heavy_atoms"] > ha_test_max) | (tr["amw"] > mw_test_max)
    ].copy()
    oversize["over_ha"] = oversize["num_heavy_atoms"] > ha_test_max
    oversize["over_mw"] = oversize["amw"] > mw_test_max
    oversize = oversize.sort_values("num_heavy_atoms", ascending=False)
    print(f"[01] train compounds larger than any test compound: {len(oversize)}")
    print(f"     potent (pEC50 >= {POTENT_PEC50}): {oversize['train_is_potent'].sum()}")
    print(
        f"     with pEC50 >= 5.5:                {(oversize['train_pec50'] >= 5.5).sum()}"
    )

    # Save flagged parquet with activity columns kept
    keep = [
        "compound_id",
        "smiles",
        "num_heavy_atoms",
        "amw",
        "num_rings",
        "logp",
        "fractioncsp3",
        "over_ha",
        "over_mw",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "train_is_potent",
        "counter_pec50",
        "counter_minus_train_pec50",
        "b2_ligand_oversize",
        "b2_pocket_distance_a",
        "pb_all_passed",
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    flagged_path = DATA_DIR.joinpath("01_size_flagged.parquet")
    oversize[keep].to_parquet(flagged_path, index=False)
    print(f"[01] wrote {flagged_path}")

    # Print top rows so we can sanity-check activity patterns
    print("\n[01] top-10 oversize train compounds (by HA), with pEC50:")
    cols_show = [
        "compound_id",
        "num_heavy_atoms",
        "amw",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "b2_pocket_distance_a",
        "pb_all_passed",
    ]
    print(oversize[cols_show].head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Figure: 4 panels
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: HA distribution train vs test
    ax = axes[0, 0]
    bins = np.arange(
        0, max(tr["num_heavy_atoms"].max(), te["num_heavy_atoms"].max()) + 3, 2
    )
    ax.hist(
        tr["num_heavy_atoms"],
        bins=bins,
        alpha=0.5,
        label=f"train (N={len(tr)})",
        color="C0",
        density=True,
    )
    ax.hist(
        te["num_heavy_atoms"],
        bins=bins,
        alpha=0.5,
        label=f"test (N={len(te)})",
        color="C1",
        density=True,
    )
    ax.axvline(
        ha_test_max,
        color="C1",
        linestyle="--",
        linewidth=1,
        label=f"test max HA = {ha_test_max:.0f}",
    )
    ax.set_xlabel("heavy atom count")
    ax.set_ylabel("density")
    ax.set_title("Heavy atom distribution")
    ax.legend()

    # Panel B: MW distribution
    ax = axes[0, 1]
    bins = np.linspace(0, max(tr["amw"].max(), te["amw"].max()) + 50, 50)
    ax.hist(tr["amw"], bins=bins, alpha=0.5, label="train", color="C0", density=True)
    ax.hist(te["amw"], bins=bins, alpha=0.5, label="test", color="C1", density=True)
    ax.axvline(
        mw_test_max,
        color="C1",
        linestyle="--",
        linewidth=1,
        label=f"test max MW = {mw_test_max:.0f}",
    )
    ax.set_xlabel("molecular weight (AMW)")
    ax.set_ylabel("density")
    ax.set_title("Molecular weight distribution")
    ax.legend()

    # Panel C: HA vs pEC50 scatter (train) with potent line + oversize highlighted
    ax = axes[1, 0]
    potent_mask = tr["train_is_potent"].fillna(False)
    ax.scatter(
        tr.loc[~potent_mask, "num_heavy_atoms"],
        tr.loc[~potent_mask, "train_pec50"],
        s=6,
        alpha=0.25,
        color="gray",
        label=f"train (pEC50 < {POTENT_PEC50})",
    )
    ax.scatter(
        tr.loc[potent_mask, "num_heavy_atoms"],
        tr.loc[potent_mask, "train_pec50"],
        s=18,
        alpha=0.9,
        color="C3",
        label=f"potent (pEC50 >= {POTENT_PEC50}, N={potent_mask.sum()})",
    )
    # Mark oversize points with edge
    if len(oversize) > 0:
        ax.scatter(
            oversize["num_heavy_atoms"],
            oversize["train_pec50"],
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=1,
            label=f"oversize vs test (N={len(oversize)})",
        )
    ax.axvline(ha_test_max, color="C1", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(POTENT_PEC50, color="C3", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("heavy atom count")
    ax.set_ylabel("train pEC50")
    ax.set_title(
        "Size vs activity (train)\nCircled points are bigger than any test compound"
    )
    ax.legend(loc="lower right", fontsize=8)

    # Panel D: MW vs pEC50 scatter (same idea)
    ax = axes[1, 1]
    ax.scatter(
        tr.loc[~potent_mask, "amw"],
        tr.loc[~potent_mask, "train_pec50"],
        s=6,
        alpha=0.25,
        color="gray",
        label="non-potent",
    )
    ax.scatter(
        tr.loc[potent_mask, "amw"],
        tr.loc[potent_mask, "train_pec50"],
        s=18,
        alpha=0.9,
        color="C3",
        label="potent",
    )
    if len(oversize) > 0:
        ax.scatter(
            oversize["amw"],
            oversize["train_pec50"],
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=1,
            label="oversize",
        )
    ax.axvline(mw_test_max, color="C1", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(POTENT_PEC50, color="C3", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("molecular weight (AMW)")
    ax.set_ylabel("train pEC50")
    ax.set_title("MW vs activity (train)")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Issue #52 01: size distribution (train vs test, with activity)", fontsize=13
    )
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_01_size_dist.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[01] wrote {fig_path}")


if __name__ == "__main__":
    main()
