"""Phase A: Filter pretrain compounds for cleaner ETKDG conformer signal.

Drops sterically complex compounds whose ETKDG conformers tend to be
high-energy / distorted, which add noise to log2fc FT gradients. Codex
2026-05-02 advice: "Phase A tests a causal hypothesis about why FT is
weak: noisy 3D supervision from junk conformers."

Filter criteria (RDKit-computable):
  - heavy_atoms <= 56  (matches Boltz-2 affinity head training cap)
  - rotatable_bonds <= 10  (rough flexibility cap)
  - max_ring_size <= 12  (excludes macrocycles)

Output: data/unimol/pretrain_log2fc_filtered.csv
        Same schema as PR #114 pretrain_labeled.csv (SMILES, log2fc_8p25,
        log2fc_33). Only the SUBSET passing filters + with at least one
        log2fc label.

Note: extract/inference keeps the FULL 13136 compounds via the existing
pretrain_all.csv. This implements Codex's "filter pretrain rows, not
extraction" hint — preserves test coverage while cleaning gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "unimol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
SELECT c.id AS compound_id,
       c.std_smiles AS smiles,
       agg.log2fc_8p25,
       agg.log2fc_33
FROM compounds c
LEFT JOIN (
  SELECT compound_id,
    AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
             THEN log2_fc_estimate END) AS log2fc_8p25,
    AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
             THEN log2_fc_estimate END) AS log2fc_33
  FROM single_concentration
  GROUP BY compound_id
) agg ON agg.compound_id = c.id
WHERE c.std_smiles IS NOT NULL
ORDER BY c.id
"""

HEAVY_CAP = 56
ROTBOND_CAP = 10
RING_CAP = 12


def passes_filter(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() > HEAVY_CAP:
        return False
    if Lipinski.NumRotatableBonds(mol) > ROTBOND_CAP:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        max_ring = max(len(r) for r in ring_info.AtomRings())
        if max_ring > RING_CAP:
            return False
    return True


def main() -> None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(SQL, conn)
    print(f"Total compounds: {len(df)}")

    # Labeled subset: BOTH heads non-null (matches PR #114 pretrain_labeled_clean.csv).
    # Newer sklearn enforces NaN check on y_pred during multi-target metric eval;
    # using the strict-both-non-null subset avoids that path entirely.
    labeled = df["log2fc_8p25"].notna() & df["log2fc_33"].notna()
    df_lab = df[labeled].copy()
    print(f"  with log2fc label: {len(df_lab)}")

    # Filter by stereochemistry / size
    keep_mask = df_lab["smiles"].apply(passes_filter)
    n_kept = int(keep_mask.sum())
    n_dropped = int((~keep_mask).sum())
    print(
        f"  filter pass: {n_kept} ({n_kept / len(df_lab) * 100:.1f}%), "
        f"dropped: {n_dropped} ({n_dropped / len(df_lab) * 100:.1f}%)"
    )

    df_out = df_lab[keep_mask].rename(columns={"smiles": "SMILES"})[
        ["SMILES", "log2fc_8p25", "log2fc_33"]
    ]

    out = OUT_DIR.joinpath("pretrain_log2fc_filtered.csv")
    df_out.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df_out)} rows × 2 heads)")

    # Sanity: also report Mol descriptor stats on dropped vs kept
    dropped = df_lab[~keep_mask]
    if len(dropped):
        print("\nDropped sample (first 5):")
        for _, row in dropped.head().iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            ha = mol.GetNumHeavyAtoms() if mol else "?"
            rb = Lipinski.NumRotatableBonds(mol) if mol else "?"
            mr = (
                max(len(r) for r in mol.GetRingInfo().AtomRings())
                if mol and mol.GetRingInfo().NumRings()
                else 0
            )
            print(f"  cid={row['compound_id']} ha={ha} rb={rb} max_ring={mr}")


if __name__ == "__main__":
    main()
