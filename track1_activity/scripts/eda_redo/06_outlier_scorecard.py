#!/usr/bin/env -S pixi run python
"""Multi-descriptor outlier scorecard.

For each of the 11 main descriptors, a compound is flagged as a tail
outlier when it sits below the train p1 or above the train p99 of that
descriptor. The scorecard then counts how many descriptors a compound
is extreme on.

Dropping compounds that are outliers on many descriptors simultaneously
is robust to the "LogP dominates" concern - they are weird on multiple
independent axes, so we are not just chasing one shortcut feature.

Two thresholds reported for each compound:
  - `n_low`  : number of descriptors where value < train p1
  - `n_high` : number of descriptors where value > train p99
  - `n_out`  : n_low + n_high
Also reports the same counts against the test [min, max] range for
cross-reference, but the drop recommendation is based on train
percentiles (does not lean on test).

Outputs:
  - eda_redo_06_outlier_scorecard.png       - histograms + top-20 list
  - 06_outlier_scorecard.parquet            - per-compound flag matrix + counts
  - 06_outlier_summary.parquet              - per-descriptor p1/p99 thresholds
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from eda_redo import load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

DESCRIPTORS = [
    "num_heavy_atoms",
    "amw",
    "logp",
    "tpsa",
    "num_rotatable_bonds",
    "hba",
    "hbd",
    "num_rings",
    "num_aromatic_rings",
    "fractioncsp3",
    "num_heteroatoms",
]

LOW_PCT = 1.0
HIGH_PCT = 99.0


def main() -> None:
    df = load_master()
    tr = df[df["split"] == "train"].copy()
    te = df[df["split"] == "test"].copy()
    print(f"[06] train N={len(tr):,}  test N={len(te):,}")

    # ------------------------------------------------------------------
    # Per-descriptor thresholds
    # ------------------------------------------------------------------
    thresh_rows = []
    for col in DESCRIPTORS:
        s = tr[col].dropna()
        thresh_rows.append(
            {
                "descriptor": col,
                "train_p1": np.percentile(s, LOW_PCT),
                "train_p99": np.percentile(s, HIGH_PCT),
                "test_min": te[col].min(),
                "test_max": te[col].max(),
                "n_train_below_p1": int((tr[col] < np.percentile(s, LOW_PCT)).sum()),
                "n_train_above_p99": int((tr[col] > np.percentile(s, HIGH_PCT)).sum()),
            }
        )
    thresholds = pd.DataFrame(thresh_rows)
    print("[06] descriptor thresholds (train p1/p99 + test range):")
    print(thresholds.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ------------------------------------------------------------------
    # Per-compound flag matrix for train
    # ------------------------------------------------------------------
    flags = pd.DataFrame(index=tr.index)
    flags["compound_id"] = tr["compound_id"].values
    for col in DESCRIPTORS:
        s = tr[col]
        p1 = np.percentile(s.dropna(), LOW_PCT)
        p99 = np.percentile(s.dropna(), HIGH_PCT)
        t_lo = te[col].min()
        t_hi = te[col].max()
        flags[f"{col}__low"] = (s < p1).fillna(False).astype(int)
        flags[f"{col}__high"] = (s > p99).fillna(False).astype(int)
        flags[f"{col}__out_test"] = ((s < t_lo) | (s > t_hi)).fillna(False).astype(int)

    low_cols = [c for c in flags.columns if c.endswith("__low")]
    high_cols = [c for c in flags.columns if c.endswith("__high")]
    out_test_cols = [c for c in flags.columns if c.endswith("__out_test")]
    flags["n_low"] = flags[low_cols].sum(axis=1)
    flags["n_high"] = flags[high_cols].sum(axis=1)
    flags["n_out"] = flags["n_low"] + flags["n_high"]
    flags["n_out_test"] = flags[out_test_cols].sum(axis=1)

    # Attach activity + key structural values for review
    ref_cols = [
        "compound_id",
        "smiles",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "num_heavy_atoms",
        "amw",
        "logp",
        "tpsa",
        "num_rotatable_bonds",
        "hba",
        "hbd",
        "num_rings",
        "num_aromatic_rings",
        "fractioncsp3",
        "num_heteroatoms",
        "b2_pocket_distance_a",
        "inchikey",
    ]
    scorecard = flags.merge(tr[ref_cols], on="compound_id", how="left")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("[06] compounds by outlier count (train p1/p99 basis):")
    dist = scorecard["n_out"].value_counts().sort_index()
    for k, v in dist.items():
        print(f"  n_out = {k:>2d}  : {v:>5d} compounds")
    print()
    print(f"  with n_out >= 3: {(scorecard['n_out'] >= 3).sum():,}")
    print(f"  with n_out >= 4: {(scorecard['n_out'] >= 4).sum():,}")
    print(f"  with n_out >= 5: {(scorecard['n_out'] >= 5).sum():,}")

    # Also against test range for reference
    print()
    print("[06] compounds by outlier count (outside test [min, max]):")
    dist_te = scorecard["n_out_test"].value_counts().sort_index()
    for k, v in dist_te.items():
        print(f"  n_out_test = {k:>2d}  : {v:>5d} compounds")

    # ------------------------------------------------------------------
    # Top-20 most-outlier compounds, show which descriptors flagged them
    # ------------------------------------------------------------------
    top = scorecard.sort_values(["n_out", "n_out_test"], ascending=[False, False]).head(
        25
    )
    print()
    print("[06] top-25 compounds by n_out (train p1/p99):")
    which_flagged_cols = low_cols + high_cols
    top_view = top.copy()
    top_view["flagged_on"] = top_view[which_flagged_cols].apply(
        lambda row: ", ".join(
            c.replace("__low", " low").replace("__high", " high")
            for c, v in row.items()
            if v == 1
        ),
        axis=1,
    )
    print(
        top_view[
            [
                "compound_id",
                "n_out",
                "n_low",
                "n_high",
                "n_out_test",
                "train_pec50",
                "num_heavy_atoms",
                "amw",
                "logp",
                "flagged_on",
            ]
        ].to_string(index=False, max_colwidth=70)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scorecard.to_parquet(DATA_DIR.joinpath("06_outlier_scorecard.parquet"), index=False)
    thresholds.to_parquet(DATA_DIR.joinpath("06_outlier_summary.parquet"), index=False)
    print()
    print("[06] wrote 06_outlier_scorecard.parquet + 06_outlier_summary.parquet")

    # ------------------------------------------------------------------
    # Figure: outlier count distribution + per-descriptor tail breakdown
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: histogram of n_out
    ax = axes[0]
    counts = scorecard["n_out"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="C0", alpha=0.8, label="train p1/p99")
    counts_te = scorecard["n_out_test"].value_counts().sort_index()
    ax.bar(
        counts_te.index + 0.35,
        counts_te.values,
        width=0.35,
        color="C1",
        alpha=0.7,
        label="outside test range",
    )
    ax.set_xlabel("number of descriptors the compound is extreme on")
    ax.set_ylabel("N train compounds")
    ax.set_title("Outlier count distribution")
    ax.set_yscale("log")
    ax.legend()

    # Panel B: how many tail-hits per descriptor (train p1 + p99 stacked)
    ax = axes[1]
    cnt_low = [flags[f"{c}__low"].sum() for c in DESCRIPTORS]
    cnt_high = [flags[f"{c}__high"].sum() for c in DESCRIPTORS]
    xs = np.arange(len(DESCRIPTORS))
    ax.bar(xs, cnt_low, color="C0", label="below p1")
    ax.bar(xs, cnt_high, bottom=cnt_low, color="C3", label="above p99")
    ax.set_xticks(xs)
    ax.set_xticklabels(DESCRIPTORS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("N train compounds")
    ax.set_title("Tail-hit count per descriptor (train p1 / p99)")
    ax.legend()

    # Panel C: n_out vs n_out_test scatter (so we can see when the two measures agree)
    ax = axes[2]
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(scorecard))
    ax.scatter(
        scorecard["n_out_test"] + jitter,
        scorecard["n_out"] + jitter,
        s=6,
        alpha=0.25,
        color="C0",
    )
    ax.plot([0, 11], [0, 11], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("n_out_test (outside test [min, max])")
    ax.set_ylabel("n_out (outside train p1/p99)")
    ax.set_title("Two outlier definitions vs each other")

    fig.suptitle(
        "Issue #52 06: multi-descriptor outlier scorecard",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_06_outlier_scorecard.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[06] wrote {fig_path}")


if __name__ == "__main__":
    main()
