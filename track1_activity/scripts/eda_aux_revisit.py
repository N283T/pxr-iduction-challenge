#!/usr/bin/env -S pixi run python
"""Revisit auxiliary data with focus on usability for the test set.

The mordred_singleconc disaster (PR #41) showed that direct concat of
counter/single-concentration features into the model does not work because
NONE of the 513 test compounds have rows in counter_assay or
single_concentration. Test compounds were ordered separately during analog
expansion and never went through the primary screen / counter screen.

This script does observation only (no model training, no DB writes). Outputs
plots + printed tables consumed by https://github.com/N283T/pxr-iduction-challenge/issues/170.

Three parts:
    Part 1: Counter-selectivity distribution and Octant's >=1.5 logunit rule
    Part 2: Single-concentration -> DRC pec50 consistency (find noisy train rows)
    Part 3: 2x2 subset (has_counter x has_singleconc) decomposition of train

Usage:
    pixi run python track1_activity/scripts/eda_aux_revisit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))
from data import get_engine  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OCTANT_SELECTIVITY_THRESHOLD = 1.5  # log units (Δ = pec50 - counter_pec50)
PRIMARY_CONC_M = 8.25e-6  # 8.25 µM is the primary screen concentration

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_all() -> dict[str, pd.DataFrame]:
    eng = get_engine()
    sql = {
        "train": """
            SELECT compound_id, pec50, pec50_std_error, emax_estimate
            FROM train_activity ORDER BY compound_id
        """,
        "test": "SELECT compound_id FROM test_activity",
        "counter": """
            SELECT compound_id, pec50 AS counter_pec50,
                   emax_estimate AS counter_emax
            FROM counter_assay
        """,
        "single_primary": f"""
            SELECT compound_id,
                   log2_fc_estimate AS log2fc,
                   log2_fc_stderr AS log2fc_se,
                   p_value, cohens_d
            FROM single_concentration
            WHERE concentration_m BETWEEN {PRIMARY_CONC_M * 0.99}
                                      AND {PRIMARY_CONC_M * 1.01}
        """,
    }
    return {k: pd.read_sql(q, eng) for k, q in sql.items()}


# ---------------------------------------------------------------------------
# Part 1: counter-selectivity
# ---------------------------------------------------------------------------


def part1_counter_selectivity(train: pd.DataFrame, counter: pd.DataFrame) -> dict:
    merged = train.merge(counter, on="compound_id", how="inner")
    merged["delta"] = merged["pec50"] - merged["counter_pec50"]
    n = len(merged)
    n_selective = int((merged["delta"] >= OCTANT_SELECTIVITY_THRESHOLD).sum())

    # Bin by pec50 to see how selectivity rate varies with potency.
    bins = [0, 4, 4.5, 5, 5.5, 6, 6.5, 7, 9]
    merged["pec50_bin"] = pd.cut(merged["pec50"], bins=bins, include_lowest=True)
    rate = (
        merged.groupby("pec50_bin", observed=True)
        .agg(
            n=("delta", "size"),
            n_selective=(
                "delta",
                lambda s: int((s >= OCTANT_SELECTIVITY_THRESHOLD).sum()),
            ),
            mean_delta=("delta", "mean"),
        )
        .assign(selective_rate=lambda d: d["n_selective"] / d["n"])
    )

    # Plot 1a: histogram of delta
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(
        merged["delta"].dropna(), bins=60, color="steelblue", edgecolor="white"
    )
    axes[0].axvline(
        OCTANT_SELECTIVITY_THRESHOLD,
        color="red",
        ls="--",
        label=f"Octant threshold = {OCTANT_SELECTIVITY_THRESHOLD}",
    )
    axes[0].set_xlabel("Δ = pec50 (PXR) − pec50 (counter)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Counter-selectivity distribution (n={n})")
    axes[0].legend()

    # Plot 1b: scatter pec50 vs delta
    sc = axes[1].scatter(
        merged["pec50"],
        merged["delta"],
        c=merged["counter_pec50"],
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    axes[1].axhline(OCTANT_SELECTIVITY_THRESHOLD, color="red", ls="--")
    axes[1].set_xlabel("PXR pec50")
    axes[1].set_ylabel("Δ (selectivity)")
    axes[1].set_title("Selectivity vs potency")
    plt.colorbar(sc, ax=axes[1], label="counter pec50")
    plt.tight_layout()
    out = FIG_DIR.joinpath("aux_revisit_part1_selectivity.png")
    plt.savefig(out, dpi=110)
    plt.close()

    return {
        "n_train_with_counter": n,
        "n_selective": n_selective,
        "frac_selective": n_selective / n,
        "rate_table": rate,
        "merged": merged,
        "fig": out,
    }


# ---------------------------------------------------------------------------
# Part 2: single-conc → DRC consistency (data quality lens)
# ---------------------------------------------------------------------------


def part2_singleconc_consistency(
    train: pd.DataFrame, single_primary: pd.DataFrame
) -> dict:
    merged = train.merge(single_primary, on="compound_id", how="inner")
    n = len(merged)
    if n == 0:
        return {"n": 0}

    # Pearson r and a simple linear fit log2fc -> pec50
    r = float(np.corrcoef(merged["log2fc"], merged["pec50"])[0, 1])
    coef = np.polyfit(merged["log2fc"], merged["pec50"], 1)
    pred = np.polyval(coef, merged["log2fc"])
    merged["resid"] = merged["pec50"] - pred
    merged["abs_resid"] = merged["resid"].abs()

    # Top noisy candidates
    noisy = merged.nlargest(20, "abs_resid")[
        ["compound_id", "pec50", "log2fc", "resid", "p_value", "cohens_d"]
    ]

    # Down-weight quantile thresholds
    q90 = float(merged["abs_resid"].quantile(0.90))
    q95 = float(merged["abs_resid"].quantile(0.95))
    n_q90 = int((merged["abs_resid"] >= q90).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(
        merged["log2fc"], merged["pec50"], s=10, alpha=0.5, color="steelblue"
    )
    xs = np.linspace(merged["log2fc"].min(), merged["log2fc"].max(), 50)
    axes[0].plot(xs, np.polyval(coef, xs), "r-", label=f"linear fit (r={r:.3f})")
    axes[0].set_xlabel(f"log2fc @ {PRIMARY_CONC_M:.2e} M (single-conc)")
    axes[0].set_ylabel("pec50 (DRC)")
    axes[0].set_title(f"Single-conc vs DRC (n={n})")
    axes[0].legend()

    axes[1].hist(merged["resid"], bins=60, color="darkorange", edgecolor="white")
    axes[1].axvline(q95, color="red", ls="--", label=f"|resid| q95={q95:.2f}")
    axes[1].axvline(-q95, color="red", ls="--")
    axes[1].set_xlabel("residual (pec50 − linear-fit)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Single-conc inconsistency residuals")
    axes[1].legend()
    plt.tight_layout()
    out = FIG_DIR.joinpath("aux_revisit_part2_consistency.png")
    plt.savefig(out, dpi=110)
    plt.close()

    return {
        "n": n,
        "pearson_r": r,
        "linear_coef": coef.tolist(),
        "resid_q90": q90,
        "resid_q95": q95,
        "n_above_q90": n_q90,
        "noisy_top20": noisy,
        "merged": merged,
        "fig": out,
    }


# ---------------------------------------------------------------------------
# Part 3: 2x2 subset decomposition
# ---------------------------------------------------------------------------


def part3_subsets(
    train: pd.DataFrame, counter: pd.DataFrame, single_primary: pd.DataFrame
) -> dict:
    has_counter = set(counter["compound_id"])
    has_single = set(single_primary["compound_id"])
    df = train.copy()
    df["has_counter"] = df["compound_id"].isin(has_counter)
    df["has_single"] = df["compound_id"].isin(has_single)

    summary = (
        df.groupby(["has_counter", "has_single"])
        .agg(
            n=("pec50", "size"),
            pec50_mean=("pec50", "mean"),
            pec50_std=("pec50", "std"),
            pec50_median=("pec50", "median"),
        )
        .reset_index()
    )

    # Plot pec50 distribution per cell
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    data = []
    for (hc, hs), grp in df.groupby(["has_counter", "has_single"]):
        labels.append(f"counter={hc}\nsingle={hs}\nn={len(grp)}")
        data.append(grp["pec50"].values)
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("pec50")
    ax.set_title(
        "Train pec50 by aux-data availability (test = counter=False, single=False)"
    )
    plt.tight_layout()
    out = FIG_DIR.joinpath("aux_revisit_part3_subsets.png")
    plt.savefig(out, dpi=110)
    plt.close()

    return {"summary": summary, "fig": out}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading tables...")
    tabs = load_all()
    for k, v in tabs.items():
        print(f"  {k}: {len(v):>6} rows")

    print("\n=== Part 1: counter-selectivity ===")
    p1 = part1_counter_selectivity(tabs["train"], tabs["counter"])
    print(f"  n_train_with_counter: {p1['n_train_with_counter']}")
    print(
        f"  n_selective (Δ>={OCTANT_SELECTIVITY_THRESHOLD}): "
        f"{p1['n_selective']} ({p1['frac_selective']:.1%})"
    )
    print("  selective rate by pec50 bin:")
    print(p1["rate_table"].to_string())
    print(f"  -> {p1['fig']}")

    print("\n=== Part 2: single-conc → DRC consistency ===")
    p2 = part2_singleconc_consistency(tabs["train"], tabs["single_primary"])
    if p2["n"] == 0:
        print("  no overlap")
    else:
        print(f"  n: {p2['n']}, pearson r: {p2['pearson_r']:.4f}")
        print(
            f"  linear: pec50 = {p2['linear_coef'][0]:.3f}*log2fc + {p2['linear_coef'][1]:.3f}"
        )
        print(
            f"  |resid| q90={p2['resid_q90']:.3f}, q95={p2['resid_q95']:.3f}, "
            f"n_above_q90={p2['n_above_q90']}"
        )
        print("  top-20 noisy compounds:")
        print(p2["noisy_top20"].to_string(index=False))
        print(f"  -> {p2['fig']}")

    print("\n=== Part 3: 2x2 subset decomposition ===")
    p3 = part3_subsets(tabs["train"], tabs["counter"], tabs["single_primary"])
    print(p3["summary"].to_string(index=False))
    print(f"  -> {p3['fig']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
