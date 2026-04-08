#!/usr/bin/env -S pixi run python
"""EDA for auxiliary data (counter-assay and single-concentration).

Addresses issue #21 — explores how counter_assay and single_concentration
data can be used to improve the Track 1 activity model.

Outputs:
    - Printed text summary
    - docs/figures/aux_data_*.png

Usage:
    pixi run python track1_activity/scripts/eda_auxiliary_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))
from data import get_engine  # noqa: E402

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_tables() -> dict[str, pd.DataFrame]:
    eng = get_engine()
    sql = {
        "train": """
            SELECT compound_id, pec50, pec50_std_error, emax_estimate, emax_vs_pos_ctrl
            FROM train_activity ORDER BY compound_id
        """,
        "counter": """
            SELECT compound_id, pec50 AS counter_pec50,
                   pec50_std_error AS counter_se,
                   emax_estimate AS counter_emax,
                   emax_vs_pos_ctrl AS counter_emax_vpc
            FROM counter_assay ORDER BY compound_id
        """,
        "test": "SELECT compound_id FROM test_activity",
        "single": """
            SELECT compound_id, concentration_m, log2_fc_estimate,
                   log2_fc_stderr, cohens_d, p_value, fdr_bh, n_replicates
            FROM single_concentration
        """,
    }
    return {k: pd.read_sql(v, eng) for k, v in sql.items()}


# ---------------------------------------------------------------------------
# Counter-assay analysis
# ---------------------------------------------------------------------------


def analyze_counter(
    train: pd.DataFrame, counter: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(" Counter-assay analysis")
    print("=" * 70)

    merged = train.merge(counter, on="compound_id", how="left")
    has_counter = merged["counter_pec50"].notna()
    both_valid = merged["pec50"].notna() & merged["counter_pec50"].notna()

    print(f"  train rows:                {len(merged)}")
    print(
        f"  train with counter pEC50:  {both_valid.sum()} ({both_valid.mean() * 100:.1f}%)"
    )
    print(f"  train without counter:     {(~has_counter).sum()}")
    print(
        f"  test ∩ counter_assay:      "
        f"{counter['compound_id'].isin(test['compound_id']).sum()} (expected 0)"
    )

    d = merged.loc[both_valid].copy()
    d["selectivity"] = d["pec50"] - d["counter_pec50"]

    print(f"\n  train pEC50:   mean={d['pec50'].mean():.3f} ± {d['pec50'].std():.3f}")
    print(
        f"  counter pEC50: mean={d['counter_pec50'].mean():.3f} ± {d['counter_pec50'].std():.3f}"
    )
    r = d[["pec50", "counter_pec50"]].corr().iloc[0, 1]
    print(f"  Pearson r(train, counter) = {r:.3f}")

    sel = d["selectivity"]
    print(f"\n  Selectivity (train − counter):")
    print(f"    mean={sel.mean():.3f}, std={sel.std():.3f}")
    qs = sel.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"    q05={qs[0.05]:.2f}  q25={qs[0.25]:.2f}  "
        f"q50={qs[0.5]:.2f}  q75={qs[0.75]:.2f}  q95={qs[0.95]:.2f}"
    )
    bins = {
        "sel < -0.5  (counter >> train, suspicious)": (sel < -0.5).sum(),
        "-0.5 ≤ sel < 0  (counter > train)": ((sel >= -0.5) & (sel < 0)).sum(),
        "0 ≤ sel < 0.3  (non-specific)": ((sel >= 0) & (sel < 0.3)).sum(),
        "0.3 ≤ sel < 1.0  (weak specificity)": ((sel >= 0.3) & (sel < 1.0)).sum(),
        "1.0 ≤ sel < 2.0  (moderate specificity)": ((sel >= 1.0) & (sel < 2.0)).sum(),
        "sel ≥ 2.0  (strong specificity)": (sel >= 2.0).sum(),
    }
    print("\n  Selectivity bins:")
    for k, v in bins.items():
        print(f"    {k:<45s} {v:5d} ({v / len(sel) * 100:5.1f}%)")

    # pEC50 × selectivity binned analysis
    print("\n  Mean pEC50 by selectivity bin:")
    d["sel_bin"] = pd.cut(
        sel,
        bins=[-np.inf, 0, 0.5, 1.0, 2.0, np.inf],
        labels=["<0", "0-0.5", "0.5-1", "1-2", ">2"],
    )
    print(
        d.groupby("sel_bin", observed=True)["pec50"]
        .agg(["count", "mean", "std"])
        .to_string()
    )

    # Figure: scatter + selectivity histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(d["counter_pec50"], d["pec50"], s=6, alpha=0.3)
    lims = [1, 8]
    axes[0].plot(lims, lims, "k--", lw=1, label="y = x")
    axes[0].plot(
        lims, [x + 1 for x in lims], "g--", lw=1, alpha=0.5, label="+1 (specific)"
    )
    axes[0].set_xlabel("counter_assay pEC50")
    axes[0].set_ylabel("train pEC50")
    axes[0].set_title(f"train vs counter (r = {r:.3f}, n = {len(d)})")
    axes[0].legend()
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)

    axes[1].hist(sel, bins=60, color="steelblue", edgecolor="k", alpha=0.7)
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].axvline(1, color="g", ls="--", lw=1)
    axes[1].set_xlabel("selectivity = pEC50 − counter_pEC50")
    axes[1].set_ylabel("count")
    axes[1].set_title("Selectivity distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR.joinpath("aux_data_counter_selectivity.png"), dpi=120)
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR.joinpath('aux_data_counter_selectivity.png')}")

    return d


# ---------------------------------------------------------------------------
# Single-concentration analysis
# ---------------------------------------------------------------------------


def analyze_single(
    train: pd.DataFrame, single: pd.DataFrame, test: pd.DataFrame
) -> None:
    print("\n" + "=" * 70)
    print(" Single-concentration analysis")
    print("=" * 70)

    n_rows = len(single)
    n_compounds = single["compound_id"].nunique()
    print(f"  total rows:           {n_rows}")
    print(f"  unique compounds:     {n_compounds}")
    print(f"  rows/compound (mean): {n_rows / n_compounds:.2f}")

    concs = single["concentration_m"].value_counts().sort_index()
    print(f"\n  Concentrations (M):")
    for c, n in concs.items():
        nc = single.loc[single["concentration_m"] == c, "compound_id"].nunique()
        print(f"    {c:.2e}  rows={n:<6} unique_compounds={nc}")

    overlap_train = single[single["compound_id"].isin(train["compound_id"])]
    print(
        f"\n  Overlap with train: {len(overlap_train)} rows, "
        f"{overlap_train['compound_id'].nunique()} compounds "
        f"({overlap_train['compound_id'].nunique() / len(train) * 100:.1f}% of train)"
    )
    print(
        f"  Overlap with test:  "
        f"{single['compound_id'].isin(test['compound_id']).sum()} rows (expected 0)"
    )

    print(f"\n  compound_class distribution:")
    print(
        single.get("compound_class", pd.Series(dtype=object))
        .value_counts(dropna=False)
        .to_string()
    )

    # Analyze at the two most populated concentrations
    main_concs = concs.sort_values(ascending=False).head(2).index.tolist()
    for conc in sorted(main_concs):
        print(f"\n  --- concentration = {conc:.2e} M ---")
        sub = single[single["concentration_m"] == conc]
        merged = train.merge(
            sub[
                [
                    "compound_id",
                    "log2_fc_estimate",
                    "log2_fc_stderr",
                    "cohens_d",
                    "p_value",
                ]
            ].rename(
                columns={
                    "log2_fc_estimate": "log2fc",
                    "log2_fc_stderr": "log2fc_se",
                }
            ),
            on="compound_id",
            how="inner",
        ).dropna(subset=["pec50", "log2fc"])
        print(f"    train ∩ conc: {len(merged)}")
        if not len(merged):
            continue
        r_log2 = merged[["pec50", "log2fc"]].corr().iloc[0, 1]
        print(f"    Pearson r(pEC50, log2fc)    = {r_log2:.3f}")
        cd = merged[["pec50", "cohens_d"]].dropna()
        if len(cd):
            r_cd = cd.corr().iloc[0, 1]
            print(f"    Pearson r(pEC50, cohens_d)  = {r_cd:.3f}  (n={len(cd)})")
        # Binned mean pEC50
        merged["log2fc_bin"] = pd.cut(
            merged["log2fc"], bins=[-np.inf, -0.5, 0, 0.5, 1, 2, np.inf]
        )
        print("    Mean pEC50 by log2fc bin:")
        print(
            merged.groupby("log2fc_bin", observed=True)["pec50"]
            .agg(["count", "mean", "std"])
            .to_string()
            .replace("\n", "\n    ")
        )

    # Aggregate features: max log2fc across concentrations per compound, slope
    print("\n  --- aggregate features across concentrations ---")
    agg = (
        single.groupby("compound_id")
        .agg(
            max_log2fc=("log2_fc_estimate", "max"),
            mean_log2fc=("log2_fc_estimate", "mean"),
            n_concs=("concentration_m", "nunique"),
        )
        .reset_index()
    )
    merged = train.merge(agg, on="compound_id", how="inner").dropna(
        subset=["pec50", "max_log2fc"]
    )
    print(f"  train ∩ single-conc (any concentration): {len(merged)}")
    if len(merged):
        r_max = merged[["pec50", "max_log2fc"]].corr().iloc[0, 1]
        r_mean = merged[["pec50", "mean_log2fc"]].corr().iloc[0, 1]
        print(f"  Pearson r(pEC50, max_log2fc)  = {r_max:.3f}")
        print(f"  Pearson r(pEC50, mean_log2fc) = {r_mean:.3f}")
        # n_concs distribution
        print("  n_concs per compound:")
        print(
            merged["n_concs"]
            .value_counts()
            .sort_index()
            .to_string()
            .replace("\n", "\n    ")
        )

    # Figure: pEC50 vs max log2fc
    if len(merged):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(merged["max_log2fc"], merged["pec50"], s=6, alpha=0.3)
        ax.set_xlabel("max log2_fc across concentrations")
        ax.set_ylabel("train pEC50")
        ax.set_title(f"pEC50 vs max log2fc (r = {r_max:.3f}, n = {len(merged)})")
        fig.tight_layout()
        fig.savefig(FIG_DIR.joinpath("aux_data_single_conc_vs_pec50.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved: {FIG_DIR.joinpath('aux_data_single_conc_vs_pec50.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tables = load_tables()
    train, counter, test, single = (
        tables["train"],
        tables["counter"],
        tables["test"],
        tables["single"],
    )

    print("=" * 70)
    print(" Auxiliary data EDA — issue #21")
    print("=" * 70)
    print(f"  train_activity:        {len(train)} rows")
    print(f"  counter_assay:         {len(counter)} rows")
    print(f"  test_activity:         {len(test)} rows")
    print(f"  single_concentration:  {len(single)} rows")

    analyze_counter(train, counter, test)
    analyze_single(train, single, test)
    analyze_triangulation(train, counter, single)

    print("\n" + "=" * 70)
    print(" Done.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Triangulation: counter-assay vs single-conc (both are off-target proxies)
# ---------------------------------------------------------------------------


def analyze_triangulation(
    train: pd.DataFrame, counter: pd.DataFrame, single: pd.DataFrame
) -> None:
    print("\n" + "=" * 70)
    print(" Triangulation: counter-assay × single-concentration × train")
    print("=" * 70)

    main_conc = 8.25e-6
    sub = single[single["concentration_m"].between(main_conc * 0.99, main_conc * 1.01)]
    sub = sub[["compound_id", "log2_fc_estimate"]].rename(
        columns={"log2_fc_estimate": "log2fc_hi"}
    )

    merged = (
        train[["compound_id", "pec50"]]
        .merge(counter[["compound_id", "counter_pec50"]], on="compound_id", how="inner")
        .merge(sub, on="compound_id", how="inner")
        .dropna()
    )
    merged["selectivity"] = merged["pec50"] - merged["counter_pec50"]
    print(f"  train ∩ counter ∩ single(8.25e-6): {len(merged)} compounds")
    if not len(merged):
        return

    corr = merged[["pec50", "counter_pec50", "log2fc_hi", "selectivity"]].corr()
    print("\n  Correlation matrix:")
    print(corr.round(3).to_string().replace("\n", "\n    "))

    print("\n  Are low-selectivity compounds also flat in single-conc?")
    merged["sel_bin"] = pd.cut(
        merged["selectivity"],
        bins=[-np.inf, 0, 0.5, 1, 2, np.inf],
        labels=["<0", "0-0.5", "0.5-1", "1-2", ">2"],
    )
    print(
        merged.groupby("sel_bin", observed=True)["log2fc_hi"]
        .agg(["count", "mean", "std"])
        .to_string()
        .replace("\n", "\n    ")
    )

    # Suspicious compounds: both counter>train AND flat single-conc
    suspicious = merged[(merged["selectivity"] < 0) & (merged["log2fc_hi"] < 0.3)]
    print(
        f"\n  Doubly-suspicious (sel<0 AND log2fc<0.3): "
        f"{len(suspicious)} / {len(merged)} ({len(suspicious) / len(merged) * 100:.1f}%)"
    )
    if len(suspicious):
        print(
            f"    their pEC50: mean={suspicious['pec50'].mean():.3f} "
            f"±{suspicious['pec50'].std():.3f}"
        )


if __name__ == "__main__":
    main()
