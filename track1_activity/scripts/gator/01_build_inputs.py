"""Convert compound_boltz2 outputs -> GatorAffinity input format.

For each compound we produce three artefacts that match the
GatorAffinity input schema (README "Custom Data Processing"):

  <OUT>/ligands/<id>_ligand.pdb   (ligand, hydrogens removed, UNL/A/1)
  <OUT>/pockets/<id>_pocket_5A.pdb (protein residues within 5 A of ligand)
  <OUT>/index.csv                  (pdb_id, paths, chain, lig_code, smiles, label)

The GatorAffinity process_pdbs.py script is then run against this CSV
from inside the fork's pixi env to produce the final .pkl feed.

Boltz-2 output conventions (constants.py):
  - protein on chain A
  - ligand on chain B, residue name LIG1 (occupies the whole chain)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402


DEFAULT_OUT = REPO_ROOT.joinpath("structures", "gator")
LIGAND_CHAIN_IN_CIF = "B"  # Boltz-2 puts ligand on chain B (LIG1)
PROTEIN_CHAIN_IN_CIF = "A"
POCKET_CUTOFF = 5.0


def write_ligand_pdb(mol: Chem.Mol, out_path: Path) -> int:
    """Write a no-H ligand PDB with resname UNL / chain A / resnum 1.

    Atom names are unique within the residue (e.g. C1, C2, N1, O1...).
    BioPython's PDBParser drops duplicates (treats same-name atoms as alt
    conformers), so GatorAffinity's fragmentation requires uniqueness.

    Returns the number of heavy atoms written.
    """
    mol_noh = Chem.RemoveHs(mol)
    counters: dict[str, int] = {}
    for atom in mol_noh.GetAtoms():
        sym = atom.GetSymbol()
        counters[sym] = counters.get(sym, 0) + 1
        raw = f"{sym}{counters[sym]}"
        # PDB atom name is 4 chars; keep within limit (rare >99 same element).
        name = raw[:4].ljust(4)
        info = Chem.AtomPDBResidueInfo(
            atomName=name,
            residueName="UNL",
            residueNumber=1,
            chainId="A",
            isHeteroAtom=True,
        )
        atom.SetMonomerInfo(info)
    Chem.MolToPDBFile(mol_noh, str(out_path), flavor=0)
    return mol_noh.GetNumAtoms()


def extract_pocket(
    cif_path: Path,
    ligand_coords: np.ndarray,
    out_path: Path,
    cutoff: float = POCKET_CUTOFF,
) -> int:
    """Write a pocket PDB containing protein residues within `cutoff` A
    of any ligand atom. Returns the residue count kept.

    Drops the ligand chain (B) and any hydrogens; keeps the protein chain
    A residues whose nearest heavy atom is within cutoff A of any ligand
    atom.
    """
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise ValueError("empty cif")
    model = st[0]
    tree = cKDTree(ligand_coords)

    # Build a fresh structure holding only the selected protein residues
    out_st = gemmi.Structure()
    out_st.name = st.name
    out_model = gemmi.Model(str(model.num))
    out_chain = gemmi.Chain(PROTEIN_CHAIN_IN_CIF)
    n_keep = 0
    for chain in model:
        if chain.name != PROTEIN_CHAIN_IN_CIF:
            continue
        for res in chain:
            if res.het_flag == "H":
                continue
            coords = np.asarray(
                [(a.pos.x, a.pos.y, a.pos.z) for a in res if a.element.name != "H"],
                dtype=np.float32,
            )
            if coords.size == 0:
                continue
            min_d = tree.query(coords, k=1)[0].min()
            if min_d < cutoff:
                res_copy = gemmi.Residue()
                res_copy.name = res.name
                res_copy.seqid = res.seqid
                res_copy.subchain = res.subchain
                res_copy.entity_type = res.entity_type
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    res_copy.add_atom(atom.clone())
                out_chain.add_residue(res_copy)
                n_keep += 1
    out_model.add_chain(out_chain)
    out_st.add_model(out_model)
    out_st.setup_entities()
    out_st.write_pdb(str(out_path))
    return n_keep


def fetch_rows(conn, subset: str) -> list[dict]:
    """Yield candidate compounds. `subset` controls selection."""
    query = """
        SELECT
            b.compound_id,
            b.pose_cif_path,
            b.ligand_pkl_path,
            c.std_smiles,
            ta.pec50
        FROM compound_boltz2 b
        JOIN compounds c ON c.id = b.compound_id
        LEFT JOIN train_activity ta ON ta.compound_id = b.compound_id
        WHERE b.preprocessing_failed = FALSE
          AND b.pose_cif_path IS NOT NULL
          AND b.ligand_pkl_path IS NOT NULL
    """
    if subset == "train":
        query += " AND ta.pec50 IS NOT NULL"
    elif subset == "smoke":
        query += " AND ta.pec50 IS NOT NULL ORDER BY b.compound_id LIMIT 20"
    elif subset == "all":
        pass
    elif subset == "test":
        query += " AND ta.pec50 IS NULL"
    else:
        raise ValueError(f"unknown subset: {subset}")
    if not subset.startswith("smoke"):
        query += " ORDER BY b.compound_id"
    df = pd.read_sql(query, conn)
    return df.to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--subset",
        default="smoke",
        choices=["smoke", "train", "test", "all"],
        help="smoke: 20 train compounds. train: 4140. test: 513. all: 4653.",
    )
    ap.add_argument("--cutoff", type=float, default=POCKET_CUTOFF)
    args = ap.parse_args()

    out_root: Path = args.out
    lig_dir = out_root.joinpath("ligands")
    pkt_dir = out_root.joinpath("pockets")
    lig_dir.mkdir(parents=True, exist_ok=True)
    pkt_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_root.joinpath(f"index_{args.subset}.csv")

    conn = psycopg2.connect(**DB_PARAMS)
    rows = fetch_rows(conn, args.subset)
    conn.close()
    print(f"Candidates: {len(rows)}  (subset={args.subset})")

    records: list[dict] = []
    n_skipped = 0
    for i, row in enumerate(rows, 1):
        cid = int(row["compound_id"])
        tag = f"{cid:05d}"
        lig_pdb = lig_dir.joinpath(f"{tag}_ligand.pdb")
        pkt_pdb = pkt_dir.joinpath(f"{tag}_pocket_5A.pdb")

        try:
            with open(row["ligand_pkl_path"], "rb") as f:
                mol = pickle.load(f)
            if not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
                raise ValueError("ligand pkl has no conformer")
            n_lig = write_ligand_pdb(mol, lig_pdb)
            lig_coords = mol.GetConformer(0).GetPositions().astype(np.float32)
            n_res = extract_pocket(
                Path(row["pose_cif_path"]),
                lig_coords,
                pkt_pdb,
                cutoff=args.cutoff,
            )
        except Exception as e:  # noqa: BLE001
            n_skipped += 1
            print(f"  skip {tag}: {type(e).__name__}: {e}")
            if lig_pdb.exists():
                lig_pdb.unlink()
            if pkt_pdb.exists():
                pkt_pdb.unlink()
            continue

        records.append(
            {
                "pdb_id": tag,
                "protein_pdb": str(pkt_pdb.resolve()),
                "ligand_pdb": str(lig_pdb.resolve()),
                "protein_chains": "A",
                "lig_code": "UNL",
                "smiles": row["std_smiles"],
                "lig_resi": 1,
                "label": row["pec50"] if row["pec50"] is not None else "",
            }
        )
        if i % 100 == 0 or i <= 20:
            print(f"  [{i}/{len(rows)}] {tag}: lig={n_lig} atoms, pkt={n_res} res")

    df = pd.DataFrame.from_records(records)
    df.to_csv(index_path, index=False)
    print(f"\nWrote {len(df)} rows -> {index_path}  (skipped={n_skipped})")


if __name__ == "__main__":
    main()
