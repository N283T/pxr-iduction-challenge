#!/usr/bin/env -S pixi run python
"""Pocket-fit (Boltz-2) EDA: pose distance and PoseBusters crossed with activity.

Every view is joined to pEC50 / Emax so that outlier poses are judged against
measured activity rather than dropped blindly.

Outputs:
  - eda_redo_02_pocket_fit.png         - 4-panel figure
  - 02_pose_outliers.parquet           - train compounds with weird poses and their pEC50
  - 02_posebusters_failures.parquet    - which checks fail most and their activity
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

from eda_redo import POTENT_PEC50, load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

PB_CHECK_COLUMNS = [
    "pb_intramol_passed",
    "pb_intermol_passed",
    "pb_no_protein_clash",
    "pb_internal_energy_ok",
]

# "far from pocket" threshold borrowed from issue #51 triage.
POCKET_DIST_OUTLIER_A = 4.0


def main() -> None:
    df = load_master()
    # Keep compounds with a Boltz-2 prediction (train + test).
    pred = df[df["b2_confidence"].notna()].copy()
    tr = pred[pred["split"] == "train"].copy()
    te = pred[pred["split"] == "test"].copy()

    print(
        f"[02] compounds with Boltz-2 prediction: {len(pred):,}"
        f"  (train {len(tr):,} + test {len(te):,})"
    )
    print()

    # ------------------------------------------------------------------
    # 1. Pocket distance distribution
    # ------------------------------------------------------------------
    for label, sub in (("train", tr), ("test", te)):
        pd_col = sub["b2_pocket_distance_a"]
        print(
            f"[02] pocket_dist ({label:5s}):  min={pd_col.min():.2f}  median={pd_col.median():.2f}"
            f"  p95={pd_col.quantile(0.95):.2f}  max={pd_col.max():.2f}"
        )

    outlier = tr[tr["b2_pocket_distance_a"] > POCKET_DIST_OUTLIER_A].copy()
    outlier = outlier.sort_values("b2_pocket_distance_a", ascending=False)
    print()
    print(f"[02] train with pocket_dist > {POCKET_DIST_OUTLIER_A} A: {len(outlier)}")
    print(
        f"     of which potent (pEC50 >= {POTENT_PEC50}): {outlier['train_is_potent'].sum()}"
    )
    print(
        f"     of which pEC50 >= 5.0:                     {(outlier['train_pec50'] >= 5.0).sum()}"
    )
    print(
        f"     of which pEC50 < 4.0:                      {(outlier['train_pec50'] < 4.0).sum()}"
    )

    keep_cols = [
        "compound_id",
        "smiles",
        "num_heavy_atoms",
        "amw",
        "b2_pocket_distance_a",
        "b2_confidence",
        "b2_ligand_iptm",
        "b2_affinity_pred",
        "b2_affinity_prob",
        "pb_all_passed",
        "pb_num_passed",
        "pb_no_protein_clash",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "train_is_potent",
        "counter_pec50",
        "counter_minus_train_pec50",
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outlier_path = DATA_DIR.joinpath("02_pose_outliers.parquet")
    outlier[keep_cols].to_parquet(outlier_path, index=False)
    print(f"[02] wrote {outlier_path}")

    print("\n[02] top-10 pose outliers (by pocket distance):")
    show_cols = [
        "compound_id",
        "b2_pocket_distance_a",
        "num_heavy_atoms",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "pb_all_passed",
    ]
    print(outlier[show_cols].head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 2. PoseBusters failure breakdown per check, with activity
    # ------------------------------------------------------------------
    rows = []
    for col in PB_CHECK_COLUMNS:
        failed = tr[tr[col] == False]  # noqa: E712 - comparing against pandas boolean
        if len(failed) == 0:
            continue
        rows.append(
            {
                "check": col,
                "n_failed_train": len(failed),
                "pec50_median": failed["train_pec50"].median(),
                "pec50_mean": failed["train_pec50"].mean(),
                "n_potent": int(failed["train_is_potent"].sum()),
                "n_with_emax_above_0p5": int(
                    (failed["train_emax_vs_pos_ctrl"] >= 0.5).sum()
                ),
            }
        )
    pb_summary = pd.DataFrame(rows).sort_values("n_failed_train", ascending=False)
    print()
    print("[02] PoseBusters check failures (train only), with activity summary:")
    print(pb_summary.to_string(index=False))
    pb_summary_path = DATA_DIR.joinpath("02_posebusters_failures.parquet")
    pb_summary.to_parquet(pb_summary_path, index=False)
    print(f"[02] wrote {pb_summary_path}")

    # ------------------------------------------------------------------
    # 3. Affinity head vs measured pEC50 (train, oversize-flagged)
    # ------------------------------------------------------------------
    affinity = tr[tr["b2_affinity_pred"].notna()].copy()
    # Align signs: Boltz-2 affinity_pred_value is higher = less active (IC50-like
    # regressor). For sanity correlation, compare to pEC50.
    corr = affinity[["b2_affinity_pred", "train_pec50"]].corr().iloc[0, 1]
    print()
    print(
        f"[02] affinity vs pEC50 pearson correlation (train, N={len(affinity)}): r = {corr:.3f}"
    )

    # ------------------------------------------------------------------
    # 4. Figure
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: pocket distance distribution
    ax = axes[0, 0]
    bins = np.linspace(
        0,
        max(tr["b2_pocket_distance_a"].max(), te["b2_pocket_distance_a"].max()) + 0.5,
        60,
    )
    ax.hist(
        tr["b2_pocket_distance_a"],
        bins=bins,
        alpha=0.5,
        color="C0",
        label=f"train (N={len(tr)})",
        density=True,
    )
    ax.hist(
        te["b2_pocket_distance_a"],
        bins=bins,
        alpha=0.5,
        color="C1",
        label=f"test (N={len(te)})",
        density=True,
    )
    ax.axvline(
        POCKET_DIST_OUTLIER_A,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"outlier cutoff = {POCKET_DIST_OUTLIER_A} A",
    )
    ax.set_xlabel("ligand - pocket centroid distance (A)")
    ax.set_ylabel("density")
    ax.set_title("Pocket-fit distribution")
    ax.legend()

    # Panel B: pocket distance vs pEC50 (train), color by Boltz-2 confidence
    ax = axes[0, 1]
    sc = ax.scatter(
        tr["b2_pocket_distance_a"],
        tr["train_pec50"],
        c=tr["b2_confidence"],
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    ax.axvline(POCKET_DIST_OUTLIER_A, color="red", linestyle="--", linewidth=1)
    ax.axhline(POTENT_PEC50, color="C3", linestyle=":", linewidth=1)
    plt.colorbar(sc, ax=ax, label="Boltz-2 confidence")
    ax.set_xlabel("pocket distance (A)")
    ax.set_ylabel("train pEC50")
    ax.set_title("Pose vs activity (train)")

    # Panel C: affinity head vs pEC50
    ax = axes[1, 0]
    ax.scatter(
        affinity["b2_affinity_pred"],
        affinity["train_pec50"],
        s=8,
        alpha=0.4,
        color="C0",
    )
    pot = affinity[affinity["train_is_potent"]]
    ax.scatter(
        pot["b2_affinity_pred"],
        pot["train_pec50"],
        s=20,
        alpha=0.9,
        color="C3",
        label=f"potent (N={len(pot)})",
    )
    ax.set_xlabel("Boltz-2 affinity_pred_value (pIC50-like)")
    ax.set_ylabel("train pEC50")
    ax.set_title(f"Boltz-2 affinity vs measured pEC50 (r={corr:.2f})")
    ax.legend()

    # Panel D: PoseBusters failure counts with potent overlay
    ax = axes[1, 1]
    xs = np.arange(len(pb_summary))
    ax.bar(
        xs, pb_summary["n_failed_train"], color="C0", alpha=0.7, label="failed train"
    )
    ax.bar(
        xs,
        pb_summary["n_potent"],
        color="C3",
        alpha=0.9,
        label=f"of which potent (pEC50 >= {POTENT_PEC50})",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [c.replace("pb_", "").replace("_passed", "") for c in pb_summary["check"]],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("N train compounds failing")
    ax.set_title("PoseBusters failures by check")
    ax.legend()

    fig.suptitle("Issue #52 02: Boltz-2 pose fit vs measured activity", fontsize=13)
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_02_pocket_fit.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] wrote {fig_path}")


if __name__ == "__main__":
    main()
