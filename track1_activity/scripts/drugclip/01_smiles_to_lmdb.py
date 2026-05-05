"""Convert SMILES list to LMDB format expected by DrugCLIP encode_mols.

DrugCLIP's `load_mols_dataset_dtwg` expects an LMDB with entries containing:
  - 'smi'         : SMILES string
  - 'atoms'       : list of element symbols (e.g. ['C', 'C', 'O'])
  - 'coordinates' : N × 3 numpy array of 3D coordinates

We generate ETKDGv3 conformers via RDKit (matches Uni-Mol pretrain default).

Usage:
    pixi run python 01_smiles_to_lmdb.py \
        --in  data/unimol/pretrain_all.csv \
        --out ~/ghq/github.com/THU-ATOM/Drug-The-Whole-Genome/data/pxr_compounds.lmdb
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def smiles_to_atoms_coords(
    smi: str, seed: int = 42
) -> tuple[list[str], np.ndarray] | None:
    """Generate ETKDGv3 conformer; return (atoms, coords) or None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    atoms: list[str] = []
    coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        atoms.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
    return atoms, coords


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_path", required=True, help="CSV with SMILES + compound_id"
    )
    ap.add_argument("--out", dest="out_path", required=True, help="LMDB output dir")
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--id-col", default="compound_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--map-size-gb", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    print(f"Loaded {len(df)} compounds from {args.in_path}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    map_size = args.map_size_gb * 1024**3
    env = lmdb.open(str(out_path), map_size=map_size, subdir=False, lock=False)

    n_ok = 0
    n_fail = 0
    fail_ids: list[int] = []
    with env.begin(write=True) as txn:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ETKDG + LMDB"):
            smi = row[args.smiles_col]
            cid = int(row[args.id_col])
            result = smiles_to_atoms_coords(smi, seed=args.seed)
            if result is None:
                n_fail += 1
                fail_ids.append(cid)
                continue
            atoms, coords = result
            entry = {"smi": smi, "atoms": atoms, "coordinates": [coords]}
            txn.put(str(cid).encode(), pickle.dumps(entry))
            n_ok += 1

    env.close()
    print(f"OK: {n_ok}, fail: {n_fail}")
    if fail_ids:
        print(f"  failed compound_ids (first 20): {fail_ids[:20]}")


if __name__ == "__main__":
    main()
