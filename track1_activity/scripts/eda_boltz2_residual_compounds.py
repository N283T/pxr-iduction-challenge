#!/usr/bin/env python
"""Render compound images for best/worst Boltz-2 predictions.

Produces 4 grids (PNG) using RDKit:
  - worst_over_predicted.png  (affinity says strong, experiment says weak)
  - worst_under_predicted.png (affinity says weak, experiment says strong)
  - best_predicted.png         (small |residual|)
  - worst_scaffold_sulfonanilide.png (focused scaffold dig)

Each tile labels compound_id, pec50, affinity_pred, residual.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

REPORT_DIR = Path(__file__).resolve().parents[1].joinpath("reports")
CSV = REPORT_DIR.joinpath("boltz2_residual_per_compound.csv")


def make_grid(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    molsPerRow: int = 4,
    sub_size: tuple[int, int] = (340, 300),
) -> None:
    mols = []
    legends = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        legends.append(
            f"id={row['compound_id']}  pEC50={row['pec50']:.2f}\n"
            f"aff={row['affinity_pred_value']:.2f}  "
            f"resid={row['residual']:+.2f}"
        )
        mols.append(mol)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=molsPerRow,
        subImgSize=sub_size,
        legends=legends,
        useSVG=False,
    )
    img.save(out_path)
    print(f"Saved: {out_path}  ({len(mols)} molecules) — {title}")


def main() -> None:
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} compounds from {CSV}")

    # 1. Worst over-predicted (Boltz says binder, experiment says weak)
    # = most negative residuals (pec50 < predicted_pec50)
    over = df.sort_values("residual").head(12)
    make_grid(
        over,
        REPORT_DIR.joinpath("boltz2_compounds_worst_over_predicted.png"),
        "Worst over-predicted: Boltz-2 too optimistic",
    )

    # 2. Worst under-predicted (Boltz says weak, experiment says strong)
    # = most positive residuals
    under = df.sort_values("residual", ascending=False).head(12)
    make_grid(
        under,
        REPORT_DIR.joinpath("boltz2_compounds_worst_under_predicted.png"),
        "Worst under-predicted: Boltz-2 too pessimistic",
    )

    # 3. Best predicted (near-zero residuals across wide pEC50 range)
    # Take extremes of pec50 with tiny residual to show full dynamic range
    best = df[df["abs_residual"] < 0.15].copy()
    bins = pd.qcut(best["pec50"], q=4, duplicates="drop")
    pick = (
        best.groupby(bins, observed=True)
        .apply(lambda g: g.sample(min(3, len(g)), random_state=0))
        .reset_index(drop=True)
    )
    make_grid(
        pick.head(12),
        REPORT_DIR.joinpath("boltz2_compounds_best_predicted.png"),
        "Best predicted across pEC50 range",
    )

    # 4. Focused look at sulfonanilide scaffold (Q4-heavy from previous analysis)
    sulfon = df[df["murcko_scaffold"] == "O=S(=O)(Nc1ccccc1)c1ccccc1"]
    sulfon_top = (
        sulfon.assign(sort_key=lambda x: x["abs_residual"])
        .sort_values("sort_key", ascending=False)
        .head(12)
    )
    if len(sulfon_top) > 0:
        make_grid(
            sulfon_top,
            REPORT_DIR.joinpath("boltz2_compounds_sulfonanilide.png"),
            f"Sulfonanilide scaffold (n={len(sulfon)}, worst {len(sulfon_top)} shown)",
        )

    # 5. Focused look at aryl-piperazine scaffold (also Q4-heavy)
    pipz = df[df["murcko_scaffold"] == "c1ccc(N2CCNCC2)cc1"]
    pipz_top = pipz.sort_values("abs_residual", ascending=False).head(12)
    if len(pipz_top) > 0:
        make_grid(
            pipz_top,
            REPORT_DIR.joinpath("boltz2_compounds_aryl_piperazine.png"),
            f"Aryl-piperazine scaffold (n={len(pipz)}, worst {len(pipz_top)} shown)",
        )

    # 6. Well-predicted scaffolds: indole and benzylamide, for contrast
    indole = df[df["murcko_scaffold"] == "c1ccc2[nH]ccc2c1"]
    indole_top = indole.sort_values("abs_residual").head(12)
    if len(indole_top) > 0:
        make_grid(
            indole_top,
            REPORT_DIR.joinpath("boltz2_compounds_indole.png"),
            f"Indole scaffold (n={len(indole)}, best {len(indole_top)} shown)",
        )


if __name__ == "__main__":
    main()
