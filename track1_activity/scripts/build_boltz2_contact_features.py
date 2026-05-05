#!/usr/bin/env -S pixi run python
"""Build residue-level Boltz-2 contact-shell features.

This is deliberately simpler than ProLIF IFPs: for each predicted pose, compute
which ligand heavy atoms sit within 4.5 A / 6.0 A of each protein residue and
aggregate by coarse ligand atom class. Missing residues are encoded as no
contact, with distance features filled to a large sentinel.
"""

from __future__ import annotations

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT.joinpath("data", "boltz2_contact_features.parquet")
DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")

ATOM_CLASSES = [
    "hydrophobe",
    "aromatic",
    "donor",
    "acceptor",
    "cation",
    "anion",
    "halogen",
]


def classify_ligand_atom(atom: Chem.Atom) -> set[str]:
    """Return coarse pharmacophore-ish classes for one ligand atom."""
    classes: set[str] = set()
    atomic_num = atom.GetAtomicNum()
    symbol = atom.GetSymbol()
    formal_charge = atom.GetFormalCharge()

    if atomic_num == 6 or symbol in {"S", "F", "Cl", "Br", "I"}:
        classes.add("hydrophobe")
    if atom.GetIsAromatic():
        classes.add("aromatic")
    if symbol in {"N", "O", "S"} and atom.GetTotalNumHs() > 0:
        classes.add("donor")
    if symbol in {"N", "O", "S", "F", "Cl", "Br", "I"} and formal_charge <= 0:
        classes.add("acceptor")
    if formal_charge > 0:
        classes.add("cation")
    if formal_charge < 0:
        classes.add("anion")
    if symbol in {"F", "Cl", "Br", "I"}:
        classes.add("halogen")
    return classes


def summarize_residue_contacts(
    residue_number: int,
    residue_coords: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_classes: list[set[str]],
    near_cutoff: float = 4.5,
    shell_cutoff: float = 6.0,
) -> dict[str, float]:
    """Summarize contacts between one residue and ligand heavy atoms."""
    if residue_coords.size == 0 or ligand_coords.size == 0:
        return {}
    diff = residue_coords[:, None, :] - ligand_coords[None, :, :]
    atom_min = np.linalg.norm(diff, axis=2).min(axis=0)
    if float(atom_min.min()) > shell_cutoff:
        return {}

    prefix = f"res{residue_number}"
    near = atom_min <= near_cutoff
    shell = atom_min <= shell_cutoff
    out: dict[str, float] = {
        f"{prefix}_min_dist": float(atom_min.min()),
        f"{prefix}_n_lig_atoms_4p5": int(near.sum()),
        f"{prefix}_n_lig_atoms_6p0": int(shell.sum()),
    }
    for atom_class in ATOM_CLASSES:
        class_mask = np.asarray(
            [atom_class in classes for classes in ligand_classes], dtype=bool
        )
        if not class_mask.any():
            continue
        class_dist = atom_min[class_mask]
        out[f"{prefix}_{atom_class}_min_dist"] = float(class_dist.min())
        out[f"{prefix}_{atom_class}_n_4p5"] = int((class_dist <= near_cutoff).sum())
        out[f"{prefix}_{atom_class}_n_6p0"] = int((class_dist <= shell_cutoff).sum())
    return out


def _protein_residue_coords(cif_path: str) -> dict[int, np.ndarray]:
    structure = gemmi.read_structure(cif_path)
    residues: dict[int, list[tuple[float, float, float]]] = {}
    for model in structure:
        chain = model["A"]
        for residue in chain:
            coords = residues.setdefault(int(residue.seqid.num), [])
            for atom in residue:
                if atom.element.name == "H":
                    continue
                coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
        break
    return {
        resno: np.asarray(coords, dtype=np.float32)
        for resno, coords in residues.items()
        if coords
    }


def _ligand_coords_and_classes(pkl_path: str) -> tuple[np.ndarray, list[set[str]]]:
    with open(pkl_path, "rb") as fh:
        mol = pickle.load(fh)
    conf = mol.GetConformer()
    coords = []
    classes = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
        classes.append(classify_ligand_atom(atom))
    return np.asarray(coords, dtype=np.float32), classes


def extract_one(task: tuple[int, str, str]) -> dict[str, float] | None:
    cid, cif_path, pkl_path = task
    try:
        ligand_coords, ligand_classes = _ligand_coords_and_classes(pkl_path)
        residues = _protein_residue_coords(cif_path)
        row: dict[str, float] = {"compound_id": int(cid)}
        for residue_number, residue_coords in residues.items():
            row.update(
                summarize_residue_contacts(
                    residue_number=residue_number,
                    residue_coords=residue_coords,
                    ligand_coords=ligand_coords,
                    ligand_classes=ligand_classes,
                )
            )
        return row
    except Exception as exc:  # noqa: BLE001
        return {"compound_id": int(cid), "_error": str(exc)}


def _load_tasks() -> list[tuple[int, str, str]]:
    sql = """
    SELECT compound_id, pose_cif_path, ligand_pkl_path
    FROM compound_boltz2
    WHERE pose_cif_path IS NOT NULL
      AND ligand_pkl_path IS NOT NULL
    ORDER BY compound_id
    """
    with psycopg2.connect(**DB_KW) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [(int(cid), str(cif), str(pkl)) for cid, cif, pkl in cur.fetchall()]


def _finalize_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("compound_id").sort_index()
    dist_cols = [c for c in df.columns if c.endswith("_min_dist")]
    count_cols = [c for c in df.columns if c not in dist_cols]
    if count_cols:
        df[count_cols] = df[count_cols].fillna(0).astype(np.int16)
    if dist_cols:
        df[dist_cols] = df[dist_cols].fillna(9.0).astype(np.float32)
    nonconstant = [c for c in df.columns if df[c].nunique(dropna=False) > 1]
    return df[nonconstant]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = _load_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(f"Tasks: {len(tasks)}")

    rows: list[dict[str, float]] = []
    errors: list[dict[str, float]] = []
    if args.workers <= 1:
        iterator = map(extract_one, tasks)
        for row in tqdm(iterator, total=len(tasks), desc="contacts"):
            if row is None:
                continue
            if "_error" in row:
                errors.append(row)
            else:
                rows.append(row)
    else:
        with Pool(args.workers) as pool:
            for row in tqdm(
                pool.imap_unordered(extract_one, tasks, chunksize=16),
                total=len(tasks),
                desc="contacts",
            ):
                if row is None:
                    continue
                if "_error" in row:
                    errors.append(row)
                else:
                    rows.append(row)

    print(f"ok={len(rows)} errors={len(errors)}")
    if errors:
        print("First errors:")
        for err in errors[:5]:
            print(err)
    df = _finalize_frame(rows)
    print(f"Feature frame: {df.shape}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
