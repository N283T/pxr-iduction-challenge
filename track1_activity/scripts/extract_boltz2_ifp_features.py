#!/usr/bin/env python
"""Extract Tier-2 ProLIF interaction fingerprints from Boltz-2 poses.

Pipeline (per compound):
  1. pose.cif -> gemmi -> protein-only PDB (chain A)
  2. PDBFixer addHs(pH=7.4)
  3. RDKit MolFromPDBFile(sanitize=False) -> ProLIF Molecule
  4. Ligand pkl -> ProLIF Molecule
  5. Fingerprint.run over (lig, prot) -> per (residue, interaction) bits

Feature space: pocket-wide (ALL 434 residues, filtered to those with any
interaction across the corpus after extraction) × 9 interaction types.
Because residue coverage varies per compound, we emit a sparse DataFrame
indexed by compound_id with ``{res}_{interaction}`` columns, 0/1 encoded.

Multiprocessing: 16 workers by default (RTX 5080 host has plenty of cores).

Output: data/boltz2_ifp_features.parquet
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import prolif as plf
import psycopg2
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT.joinpath("data", "boltz2_ifp_features.parquet")
DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")

# Keep these interactions; drop rare/niche ones for PXR (metal/halogen,
# separate pi stacking modes already covered by PiStacking alias).
INTERACTIONS = [
    "Hydrophobic",
    "HBDonor",
    "HBAcceptor",
    "Anionic",
    "Cationic",
    "PiCation",
    "CationPi",
    "PiStacking",
    "VdWContact",
]


def extract_one(args: tuple[int, str, str]) -> dict | None:
    cid, cif_path, pkl_path = args
    try:
        # 1. Protein-only PDB via gemmi
        st = gemmi.read_structure(cif_path)
        for model in st:
            for c in list(model):
                if c.name != "A":
                    model.remove_chain(c.name)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            st.write_pdb(f.name)
            raw_pdb = f.name

        # 2. addHs via PDBFixer
        fixer = PDBFixer(filename=raw_pdb)
        fixer.addMissingHydrogens(7.4)
        prot_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
        with open(prot_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        # 3. Protein via RDKit (sanitize=False to tolerate PDBFixer artefacts)
        prot_rdkit = Chem.MolFromPDBFile(
            prot_pdb, removeHs=False, sanitize=False, proximityBonding=True
        )
        if prot_rdkit is None:
            return None
        prot_mol = plf.Molecule(prot_rdkit)

        # 4. Ligand from pkl (bond info intact from post-processing)
        with open(pkl_path, "rb") as fh:
            lig = pickle.load(fh)
        lig_mol = plf.Molecule.from_rdkit(lig)

        # 5. Fingerprint
        fp = plf.Fingerprint(interactions=INTERACTIONS)
        fp.run_from_iterable([lig_mol], prot_mol, progress=False)
        df = fp.to_dataframe()
        # df has a MultiIndex (ligand, protein, interaction); one row only
        if df.empty:
            return {"compound_id": cid}
        row = df.iloc[0]  # first (only) frame
        out = {"compound_id": cid}
        for (_lig_name, protein_res, interaction), val in row.items():
            if val:
                out[f"{protein_res}_{interaction}"] = 1
        return out
    except Exception as e:  # noqa: BLE001
        return {"compound_id": cid, "_error": str(e)}


def main() -> None:
    with psycopg2.connect(**DB_KW) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT compound_id, pose_cif_path, ligand_pkl_path
            FROM compound_boltz2
            WHERE pose_cif_path IS NOT NULL
              AND ligand_pkl_path IS NOT NULL
            ORDER BY compound_id
            """
        )
        tasks = cur.fetchall()
    print(f"Tasks: {len(tasks)} compounds")

    # Single-process sanity run on first task
    t0 = time.time()
    first = extract_one(tasks[0])
    print(f"First compound ok: {first.get('compound_id')} in {time.time() - t0:.1f}s")

    n_workers = 16
    results: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()
    with Pool(n_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(extract_one, tasks, chunksize=8),
            total=len(tasks),
            desc="ProLIF",
        ):
            if res is None:
                continue
            if "_error" in res:
                errors.append(res)
            else:
                results.append(res)
    print(
        f"\nDone in {(time.time() - t0) / 60:.1f} min. "
        f"ok={len(results)} err={len(errors)}"
    )
    if errors:
        print("First 3 errors:")
        for e in errors[:3]:
            print(" ", e)

    df = pd.DataFrame(results).set_index("compound_id").sort_index()
    df = df.fillna(0).astype(np.int8)
    print(f"IFP DataFrame shape: {df.shape}")
    # Drop empty columns (interactions that never fire in the whole corpus)
    nonzero = df.columns[df.sum(axis=0) > 0]
    df = df[nonzero]
    print(f"After dropping all-zero columns: {df.shape}")
    print("Column prevalence (top 20):")
    print((df.sum(axis=0).sort_values(ascending=False).head(20) / len(df)).round(3))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
