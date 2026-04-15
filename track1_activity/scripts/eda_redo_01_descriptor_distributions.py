#!/usr/bin/env -S pixi run python
"""Descriptor distribution comparison: train vs test, with potent overlay.

Before calling anything an outlier, look at the raw distributions.
Each panel shows:
  - train density (grey)
  - test density (orange)
  - train potent density (red, pEC50 >= 6)
  - dashed lines at test min / max
so we can see at a glance:
  - which descriptors have train tails reaching far beyond test
  - whether potent compounds cluster in a specific part of the range
  - whether test itself is shifted relative to the bulk of train

Outputs:
  - eda_redo_01_descriptor_panels.png   - 3x4 grid of KDE panels (one per descriptor)
  - 01_descriptor_summary.parquet       - per-descriptor stats (train/test/potent ranges + KS-ish gap)
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

# (column, pretty label, suggested bin count, log-scale y?)
DESCRIPTORS = [
    ("num_heavy_atoms", "Heavy atom count", 40, False),
    ("amw", "Molecular weight (AMW)", 40, False),
    ("logp", "logP", 40, False),
    ("tpsa", "TPSA", 40, False),
    ("num_rotatable_bonds", "Rotatable bonds", 25, False),
    ("hba", "H-bond acceptors (HBA)", 25, False),
    ("hbd", "H-bond donors (HBD)", 15, False),
    ("num_rings", "Rings", 16, False),
    ("num_aromatic_rings", "Aromatic rings", 8, False),
    ("fractioncsp3", "Fraction sp3", 30, False),
    ("num_heteroatoms", "Heteroatoms", 30, False),
]


def _range_str(s: pd.Series) -> str:
    s = s.dropna()
    return f"{s.min():.2f} - {s.max():.2f} (med {s.median():.2f})"


def _panel(ax, col: str, label: str, bins: int, tr: pd.DataFrame, te: pd.DataFrame,
           potent: pd.DataFrame) -> None:
    all_vals = pd.concat([tr[col], te[col]]).dropna()
    lo, hi = all_vals.min(), all_vals.max()
    # Small padding so the test bracket is visible
    span = max(hi - lo, 1e-6)
    edges = np.linspace(lo - 0.02 * span, hi + 0.02 * span, bins + 1)

    ax.hist(tr[col].dropna(), bins=edges, color="grey", alpha=0.45,
            density=True, label=f"train (N={tr[col].notna().sum()})")
    ax.hist(te[col].dropna(), bins=edges, color="C1", alpha=0.55,
            density=True, label=f"test (N={te[col].notna().sum()})")
    if len(potent) and potent[col].notna().any():
        ax.hist(potent[col].dropna(), bins=edges, color="C3", alpha=0.85,
                density=True, histtype="step", linewidth=1.6,
                label=f"potent (N={potent[col].notna().sum()})")

    t_lo = te[col].min()
    t_hi = te[col].max()
    ax.axvline(t_lo, color="C1", linestyle=":", linewidth=1)
    ax.axvline(t_hi, color="C1", linestyle=":", linewidth=1)

    # Annotate what % of train falls outside the test range
    below = (tr[col] < t_lo).sum()
    above = (tr[col] > t_hi).sum()
    out_pct = (below + above) / max(tr[col].notna().sum(), 1) * 100
    ax.set_title(f"{label}\ntrain outside test range: {below}+{above} ({out_pct:.1f}%)", fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    ax.legend(fontsize=7, loc="best")


def main() -> None:
    df = load_master()
    tr = df[df["split"] == "train"].copy()
    te = df[df["split"] == "test"].copy()
    potent = tr[tr["train_is_potent"].fillna(False)]
    print(f"[01] train N={len(tr):,}  test N={len(te):,}  potent N={len(potent):,}")

    # ------------------------------------------------------------------
    # Per-descriptor stat table
    # ------------------------------------------------------------------
    stats_rows = []
    for col, label, _bins, _log in DESCRIPTORS:
        t_lo, t_hi = te[col].min(), te[col].max()
        below = int((tr[col] < t_lo).sum())
        above = int((tr[col] > t_hi).sum())
        stats_rows.append({
            "descriptor": col,
            "train_range": _range_str(tr[col]),
            "test_range": _range_str(te[col]),
            "potent_range": _range_str(potent[col]),
            "train_below_test_min": below,
            "train_above_test_max": above,
            "train_outside_pct": (below + above) / max(tr[col].notna().sum(), 1) * 100.0,
            "potent_outside_test_range": int(
                ((potent[col] < t_lo) | (potent[col] > t_hi)).sum()
            ),
        })
    stats = pd.DataFrame(stats_rows)
    print()
    print(stats.to_string(index=False))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = DATA_DIR.joinpath("01_descriptor_summary.parquet")
    stats.to_parquet(stats_path, index=False)
    print(f"\n[01] wrote {stats_path}")

    # ------------------------------------------------------------------
    # Figure: 3 x 4 grid (one extra slot empty)
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = len(DESCRIPTORS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for ax, (col, label, bins, _log) in zip(axes, DESCRIPTORS):
        _panel(ax, col, label, bins, tr, te, potent)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "Issue #52 01: descriptor distributions (train grey / test orange / potent red)",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_01_descriptor_panels.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[01] wrote {fig_path}")


if __name__ == "__main__":
    main()
