# PXR Structural Data

Structural data for Pregnane X Receptor (PXR, UniProt O75469) preparation
for Track 2 and structure-based features in Track 1.

See [issue #13](https://github.com/N283T/pxr-iduction-challenge/issues/13) for
the full analysis (pocket residues, clustering, ligand binding sites) and
[issue #22](https://github.com/N283T/pxr-iduction-challenge/issues/22) for
CCD-vs-challenge dataset comparison.

## Contents

### Raw Structures
- `pxr_lbd/` -- 72 PXR-LBD crystal structures from PDB (NextGen mmCIF, gzipped)
- `alphafold/` -- AlphaFold DB v6 prediction `AF-O75469-F1-model_v6.cif.gz`
  (full-length, 434 residues)

### AF-Aligned Multi-Model CIFs
- `aligned/PXR_all_protein_only.cif` -- 73 models (AF + 72 PDB), chain A only,
  water/solvent/ligands removed. Each model contains one chain named with the
  PDB ID (or `AF`).
- `aligned_with_ligands/PXR_all_with_ligands.cif` -- Same as above but
  ligands retained (water and common solvents removed).

Both are aligned to AlphaFold chain A via ChimeraX `matchmaker` (Cα-based,
Needleman-Wunsch + iterative pruning).

### Metadata
- `pxr_structure_info.json` -- Per-structure metadata: chains, homodimer
  flag, ligand CCD IDs, pocket residue list
- `pxr_pocket_residues.json` -- Binding pocket residues derived from ligand
  contacts (4.5 Å cutoff) across all holo structures. Includes core (≥50
  structures) and extended (≥10) sets. Residue numbering follows UniProt
  O75469.
- `pxr_ccd_ligands.csv` -- SMILES and names of 55 unique CCD ligands from
  the PDB co-crystal structures (from `pmb query` against the cc schema)
- `pxr_vs_alphafold_rmsd.csv` -- Full-chain Cα / heavy atom RMSD of each PDB
  structure to AlphaFold
- `pxr_pocket_rmsd_to_af.csv` -- Pocket-only heavy atom RMSD to AlphaFold

## Key Findings

- **AF vs PDB agreement**: Cα RMSD 0.44 Å vs 8SVN apo. AF is a valid
  template for docking.
- **Single orthosteric pocket**: 65/72 chain A ligands bind within 4 Å of
  the mean binding site centroid. Outliers (7) are all estradiol or
  pesticide co-binding events in the coactivator-groove adjacent site.
- **Core pocket (13 residues)**: 209, 211, 240, 243, 247, 281, 285, 288,
  299, 306, 323, 407, 411
- **Pocket conformational clusters** (1.2 Å threshold): 7 clusters,
  dominated by one main cluster (56/72 including AF)
- **CCD vs challenge dataset**: 4 exact matches in train (RFP, SRL, 3WF,
  444), no exact matches in test. Max Tanimoto similarity to test
  compounds is 0.33. PDB structures serve as **receptor templates**, not
  ligand templates.
