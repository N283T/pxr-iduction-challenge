#!/usr/bin/env -S pixi run python
"""Activity cross-check: counter-assay selectivity and single-conc agreement.

Purpose:
  - Does a high train pEC50 correspond to PXR-specific activity, or could it
    be general reporter-assay promiscuity? counter_assay is the PXR-null
    control; a compound with counter_pec50 ~= train_pec50 is likely
    non-selective / artifact.
  - Does the single-concentration screen agree with the dose-response
    pEC50? Disagreement flags potential dose-response curve issues.

Outputs:
  - eda_redo_04_activity_cross.png     - 4-panel figure
  - 04_non_selective.parquet            - train compounds whose counter_pec50
                                           is close to (or exceeds) train_pec50
  - 04_single_disagree.parquet          - compounds where dose-response and
                                           single-conc tell different stories
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

# If |counter_pec50 - train_pec50| < this, the reporter may be responding
# to the compound rather than PXR specifically.
SELECTIVITY_WINDOW = 0.5


def main() -> None:
    df = load_master()
    tr = df[df["split"] == "train"].copy()
    print(f"[04] train N = {len(tr):,}")

    # ------------------------------------------------------------------
    # 1. Counter-assay selectivity
    # ------------------------------------------------------------------
    counter = tr[tr["counter_pec50"].notna()].copy()
    print(
        f"[04] train compounds with counter_assay: {len(counter):,}  ({len(counter) / len(tr):.0%})"
    )
    diff = counter["counter_pec50"] - counter["train_pec50"]
    print(
        f"[04] counter - train pEC50:  median={diff.median():.2f}  mean={diff.mean():.2f}"
        f"  p05={diff.quantile(0.05):.2f}  p95={diff.quantile(0.95):.2f}"
    )

    non_selective = counter[diff.abs() < SELECTIVITY_WINDOW].copy()
    print(
        f"[04] non-selective (|counter - train| < {SELECTIVITY_WINDOW}): {len(non_selective):,}"
        f"  ({len(non_selective) / len(counter):.0%})"
    )
    # High-activity non-selective is the most dangerous: potent PXR-looking
    # signal that may actually be reporter artifact.
    hi_ns = non_selective[non_selective["train_pec50"] >= 5.0]
    print(
        f"[04] of which pEC50 >= 5.0 (potentially artifact-driven 'hits'): {len(hi_ns):,}"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ns_cols = [
        "compound_id",
        "smiles",
        "train_pec50",
        "counter_pec50",
        "train_emax_vs_pos_ctrl",
        "counter_emax_vs_pos_ctrl",
        "counter_minus_train_pec50",
        "train_is_potent",
        "b2_pocket_distance_a",
        "b2_affinity_pred",
    ]
    ns_path = DATA_DIR.joinpath("04_non_selective.parquet")
    non_selective[ns_cols].sort_values("train_pec50", ascending=False).to_parquet(
        ns_path, index=False
    )
    print(f"[04] wrote {ns_path}")

    # ------------------------------------------------------------------
    # 2. Single-concentration agreement
    # ------------------------------------------------------------------
    single = tr[tr["single_max_log2_fc"].notna()].copy()
    print()
    print(
        f"[04] train compounds with single_concentration data: {len(single):,}  ({len(single) / len(tr):.0%})"
    )
    # Spearman because single_max_log2_fc and pec50 are on different scales
    s_corr = (
        single[["single_max_log2_fc", "train_pec50"]].corr(method="spearman").iloc[0, 1]
    )
    print(f"[04] single_max_log2_fc vs train_pec50 spearman: rho = {s_corr:.3f}")

    # Disagreement: train potent but single says inactive (or vice versa)
    disagree_hi = single[
        (single["train_is_potent"].fillna(False))
        & (single["single_max_log2_fc"] < single["single_max_log2_fc"].median())
    ]
    disagree_lo = single[
        (single["train_pec50"] < 4.0)
        & (single["single_max_log2_fc"] > single["single_max_log2_fc"].quantile(0.95))
    ]
    print(f"[04] disagreement - potent train but low single: {len(disagree_hi):,}")
    print(f"[04] disagreement - low train but very high single: {len(disagree_lo):,}")
    disagree_cols = [
        "compound_id",
        "smiles",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "single_max_log2_fc",
        "single_max_neg_log10_fdr",
        "single_dominant_class",
        "train_is_potent",
    ]
    disagree_path = DATA_DIR.joinpath("04_single_disagree.parquet")
    import pandas as pd

    pd.concat(
        [
            disagree_hi.assign(case="potent_train_low_single"),
            disagree_lo.assign(case="low_train_high_single"),
        ]
    )[disagree_cols + ["case"]].to_parquet(disagree_path, index=False)
    print(f"[04] wrote {disagree_path}")

    # ------------------------------------------------------------------
    # 3. Emax inspection: low pEC50 + high Emax == 'noisy' curve?
    # ------------------------------------------------------------------
    print()
    print("[04] train Emax vs pEC50 diagnostics:")
    for bucket, sub in [
        ("pEC50 < 3", tr[tr["train_pec50"] < 3]),
        ("3 <= pEC50 < 4", tr[(tr["train_pec50"] >= 3) & (tr["train_pec50"] < 4)]),
        ("4 <= pEC50 < 5", tr[(tr["train_pec50"] >= 4) & (tr["train_pec50"] < 5)]),
        ("5 <= pEC50 < 6", tr[(tr["train_pec50"] >= 5) & (tr["train_pec50"] < 6)]),
        (f"pEC50 >= {POTENT_PEC50}", tr[tr["train_pec50"] >= POTENT_PEC50]),
    ]:
        if len(sub) == 0:
            continue
        print(
            f"  {bucket:<25s} N={len(sub):>5d}"
            f"  Emax(median)={sub['train_emax_vs_pos_ctrl'].median():.2f}"
            f"  Emax(p95)={sub['train_emax_vs_pos_ctrl'].quantile(0.95):.2f}"
        )

    # ------------------------------------------------------------------
    # 4. Figure: 4 panels
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: train vs counter pEC50 (selectivity map)
    ax = axes[0, 0]
    ax.scatter(
        counter["train_pec50"],
        counter["counter_pec50"],
        s=8,
        alpha=0.35,
        color="C0",
        label=f"train (N={len(counter)})",
    )
    pot = counter[counter["train_is_potent"].fillna(False)]
    ax.scatter(
        pot["train_pec50"],
        pot["counter_pec50"],
        s=30,
        alpha=0.95,
        color="C3",
        label=f"potent (pEC50 >= {POTENT_PEC50})",
    )
    # y = x line (identity)
    lo = min(counter["train_pec50"].min(), counter["counter_pec50"].min())
    hi = max(counter["train_pec50"].max(), counter["counter_pec50"].max())
    ax.plot(
        [lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1, label="y = x"
    )
    # selectivity window
    ax.plot(
        [lo, hi],
        [lo - SELECTIVITY_WINDOW, hi - SELECTIVITY_WINDOW],
        linestyle=":",
        color="red",
        linewidth=1,
        alpha=0.6,
    )
    ax.plot(
        [lo, hi],
        [lo + SELECTIVITY_WINDOW, hi + SELECTIVITY_WINDOW],
        linestyle=":",
        color="red",
        linewidth=1,
        alpha=0.6,
        label=f"|diff| = {SELECTIVITY_WINDOW} (non-selective)",
    )
    ax.set_xlabel("train pEC50 (PXR)")
    ax.set_ylabel("counter pEC50 (PXR-null)")
    ax.set_title("Counter-assay selectivity")
    ax.legend(loc="best", fontsize=8)

    # Panel B: train pEC50 vs single_max_log2_fc
    ax = axes[0, 1]
    ax.scatter(
        single["train_pec50"],
        single["single_max_log2_fc"],
        s=6,
        alpha=0.3,
        color="C0",
    )
    pot_s = single[single["train_is_potent"].fillna(False)]
    ax.scatter(
        pot_s["train_pec50"],
        pot_s["single_max_log2_fc"],
        s=25,
        alpha=0.9,
        color="C3",
        label="potent",
    )
    ax.axvline(POTENT_PEC50, color="C3", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("train pEC50")
    ax.set_ylabel("single-conc max log2 fold change")
    ax.set_title(f"Single-conc vs dose-response (spearman rho={s_corr:.2f})")
    ax.legend()

    # Panel C: train_pec50 distribution by counter-assay status
    ax = axes[1, 0]
    selective = counter[diff.abs() >= SELECTIVITY_WINDOW]
    ns_hist = non_selective["train_pec50"].dropna()
    sel_hist = selective["train_pec50"].dropna()
    bins = np.linspace(2, 8, 30)
    ax.hist(
        sel_hist,
        bins=bins,
        alpha=0.6,
        color="C0",
        label=f"selective (N={len(sel_hist)})",
        density=True,
    )
    ax.hist(
        ns_hist,
        bins=bins,
        alpha=0.6,
        color="C3",
        label=f"non-selective (N={len(ns_hist)})",
        density=True,
    )
    ax.axvline(POTENT_PEC50, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("train pEC50")
    ax.set_ylabel("density")
    ax.set_title("pEC50 distribution by selectivity")
    ax.legend()

    # Panel D: pEC50 vs Emax (curve quality check)
    ax = axes[1, 1]
    ax.scatter(
        tr["train_pec50"],
        tr["train_emax_vs_pos_ctrl"],
        s=6,
        alpha=0.3,
        color="C0",
    )
    pot_e = tr[tr["train_is_potent"].fillna(False)]
    ax.scatter(
        pot_e["train_pec50"],
        pot_e["train_emax_vs_pos_ctrl"],
        s=25,
        alpha=0.9,
        color="C3",
        label="potent",
    )
    ax.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Emax = positive control",
    )
    ax.set_xlabel("train pEC50")
    ax.set_ylabel("train Emax vs pos ctrl")
    ax.set_title("Curve quality sanity check")
    ax.legend()

    fig.suptitle(
        "Issue #52 04: activity cross-check (selectivity + single-conc)", fontsize=13
    )
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_04_activity_cross.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[04] wrote {fig_path}")


if __name__ == "__main__":
    main()
