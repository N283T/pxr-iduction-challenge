#!/usr/bin/env python
"""Summarize feature importance from the final bake-off model.

Reads boltz2_ifp_bakeoff_importance_final.csv (mordred+tier0+tier1+tier2).
Categorizes each feature into Mordred / Tier0 / Tier1 / Tier2 and reports:
  - Aggregate gain share per tier
  - Top N features overall
  - Top N within each tier
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT.joinpath("reports", "boltz2_ifp_bakeoff_importance_final.csv")

TIER0_COLS = {
    "affinity_pred_value", "affinity_pred_value_1", "affinity_pred_value_2",
    "affinity_probability_binary", "affinity_probability_binary_1",
    "affinity_probability_binary_2",
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    "ligand_atom_count", "ligand_to_pocket_distance_a",
    "ensemble_diff_affinity", "ensemble_diff_prob",
}

TIER1_PREFIXES = (
    "plddt_", "pae_", "pde_",  # aggregated stats
)


def classify(feat: str) -> str:
    if feat in TIER0_COLS:
        return "tier0_scalar"
    # tier-1 stats: plddt_*, pae_*, pde_* (covers per-residue too: plddt_res407)
    if any(feat.startswith(p) for p in TIER1_PREFIXES):
        return "tier1_confmap"
    # tier-2 IFP: "RESNAME123.A_Interaction"
    if ".A_" in feat:
        return "tier2_ifp"
    return "mordred"


def main() -> None:
    df = pd.read_csv(CSV)
    df["tier"] = df["feature"].map(classify)

    # Aggregate gain per tier
    agg = (
        df.groupby("tier")
        .agg(n_features=("feature", "count"),
             gain_sum=("gain_sum", "sum"),
             gain_max=("gain_sum", "max"),
             nonzero=("gain_sum", lambda s: (s > 0).sum()))
        .sort_values("gain_sum", ascending=False)
    )
    total_gain = df["gain_sum"].sum()
    agg["gain_share_%"] = (agg["gain_sum"] / total_gain * 100).round(2)
    agg["nonzero_frac"] = (agg["nonzero"] / agg["n_features"]).round(3)
    print("=== Gain share by tier ===")
    print(agg.to_string())

    print("\n=== Top 30 features overall ===")
    print(
        df.sort_values("gain_sum", ascending=False)
        .head(30)[["feature", "tier", "gain_sum", "rank"]]
        .to_string(index=False)
    )

    for tier in ["tier0_scalar", "tier1_confmap", "tier2_ifp", "mordred"]:
        sub = df[df["tier"] == tier].sort_values("gain_sum", ascending=False)
        top = min(15, (sub["gain_sum"] > 0).sum())
        print(f"\n=== Top {top} {tier} features (of {len(sub)}, "
              f"{(sub['gain_sum'] > 0).sum()} with gain>0) ===")
        print(sub.head(top)[["feature", "gain_sum", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
