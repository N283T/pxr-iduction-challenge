#!/usr/bin/env -S pixi run python
"""Build a per-holo PXR ligand template database.

For each PDB structure in ``structures/pxr_lbd/`` that has a ligand:
  1. Identify all ligand residue instances (CCD codes from
     ``pxr_structure_info.json`` and ``pxr_ccd_ligands.csv``).
  2. Download the RCSB ideal SDF for each unique CCD code (cached).
  3. Replace the ideal coordinates with the experimental heavy-atom
     coordinates from the holo CIF (matched by atom name), drop the
     ideal hydrogens (X-ray refs don't have them and their idealised
     positions would no longer be self-consistent after the heavy-atom
     swap), and save a per-holo SDF.

The output SDFs live under
``structures/pxr_lbd/holo_ligands_aligned/<pdb_id>_<ccd>_<instance>.sdf``.
A summary CSV at ``docs/track2/track2_holo_ligand_db.csv`` lists
(pdb_id, ccd, instance_idx, smiles, sdf_path, n_heavy_atoms).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))

import pandas as pd  # noqa: E402

from track2.constants import REPO_ROOT as PROJECT_ROOT  # noqa: E402

PXR_LBD_DIR = PROJECT_ROOT.joinpath("structures", "pxr_lbd")
ALIGNED_CIF = PROJECT_ROOT.joinpath(
    "structures", "aligned_with_ligands", "PXR_all_with_ligands.cif"
)
IDEAL_SDF_CACHE = PXR_LBD_DIR.joinpath("ideal_sdf_cache")
HOLO_SDF_DIR = PXR_LBD_DIR.joinpath("holo_ligands_aligned")
STRUCTURE_INFO = PROJECT_ROOT.joinpath("structures", "pxr_structure_info.json")
CCD_CSV = PROJECT_ROOT.joinpath("structures", "pxr_ccd_ligands.csv")
SUMMARY_CSV = PROJECT_ROOT.joinpath("docs", "track2", "track2_holo_ligand_db.csv")

RCSB_IDEAL_URL = "https://files.rcsb.org/ligands/download/{ccd}_ideal.sdf"


def _download_ideal_sdf(ccd: str) -> Path:
    """Download (or reuse cached) RCSB CCD ideal SDF for a ligand code."""
    IDEAL_SDF_CACHE.mkdir(parents=True, exist_ok=True)
    out = IDEAL_SDF_CACHE.joinpath(f"{ccd}.sdf")
    if out.exists() and out.stat().st_size > 0:
        return out
    url = RCSB_IDEAL_URL.format(ccd=ccd)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to download {url}: {exc}")
    out.write_bytes(data)
    return out


def _load_ideal_heavy_mol(ccd: str):
    """Return RDKit Mol for the CCD with hydrogens removed but bonds intact."""
    from rdkit import Chem  # noqa: PLC0415

    sdf_path = _download_ideal_sdf(ccd)
    sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols = [m for m in sup if m is not None]
    if not mols:
        raise RuntimeError(f"could not parse {sdf_path}")
    mol = mols[0]
    # Capture original atom names (RDKit keeps them under the AtomPDBResidueInfo
    # for SDFs that have explicit atom-name properties — RCSB ideal SDFs use
    # the "molFileAlias" atom-level property in the V2000 atom block to encode
    # the CCD atom name when available. RCSB sometimes emits them via the
    # extended atom block; we recover them robustly by reading the SDF text
    # directly because RDKit's SDF parser does NOT preserve the CCD atom
    # name property by default.
    return mol, sdf_path


def _read_atom_names_from_sdf(sdf_path: Path) -> list[str | None]:
    """Read atom names from the SDF V2000 atom block.

    RCSB CCD ideal SDFs include the atom name as the first three characters
    of each atom line in the atom block. RDKit does not surface these by
    default. We parse the V2000 block directly.
    """
    text = sdf_path.read_text()
    lines = text.splitlines()
    # V2000 counts line is the 4th line: "  N  M  ..."
    if len(lines) < 4:
        return []
    counts = lines[3]
    try:
        n_atoms = int(counts[:3].strip())
    except ValueError:
        return []
    names: list[str | None] = []
    for _ in range(n_atoms):
        # The element symbol field is at columns 31-34 in V2000.
        # RCSB CCD ideal SDFs do NOT include atom names in the standard atom
        # line. The atom names live in an "M  ALS" or appended properties
        # block, OR are absent entirely. Fall back to None and rely on
        # element-based matching as a secondary path.
        names.append(None)
    return names


def _extract_holo_ligands(structure_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk every per-PDB CIF and yield one entry per ligand residue.

    Uses the per-PDB ``structures/pxr_lbd/pdb_<id>_xyz-enrich.cif.gz`` files
    rather than the aligned multi-model CIF: the aligned CIF flattens
    multi-instance HETATM residues (e.g. SRL × 3 in 1ILH) into a single
    99-atom residue, which breaks element-order matching to the ideal
    SDF. Per-PDB CIFs preserve chain structure so each ligand instance
    stays separate.

    Coordinates are returned in the per-PDB original frame; the
    template-transfer script aligns the Boltz protein to the relevant
    per-PDB protein chain at use time.
    """
    import gemmi  # noqa: PLC0415

    out: list[dict[str, Any]] = []
    for entry in structure_info:
        pdb = entry["pdb_id"]
        if not entry["ligands"]:
            continue
        cif_path = PXR_LBD_DIR.joinpath(f"pdb_0000{pdb.lower()}_xyz-enrich.cif.gz")
        if not cif_path.exists():
            continue
        struct = gemmi.read_structure(str(cif_path))
        # Per-PDB CIFs typically have a single model.
        if len(struct) == 0:
            continue
        model = struct[0]
        instance_counter: dict[str, int] = {}
        for chain in model:
            for res in chain:
                if res.het_flag != "H" or res.name == "HOH":
                    continue
                ccd = res.name
                idx = instance_counter.get(ccd, 0)
                instance_counter[ccd] = idx + 1
                atoms = [
                    {
                        "name": atom.name,
                        "element": atom.element.name,
                        "x": atom.pos.x,
                        "y": atom.pos.y,
                        "z": atom.pos.z,
                    }
                    for atom in res
                    if atom.element.atomic_number > 1  # skip H
                ]
                out.append(
                    {
                        "pdb_id": pdb,
                        "ccd": ccd,
                        "instance": idx,
                        "chain_id": chain.name,
                        "residue_seqid": res.seqid.num,
                        "atoms": atoms,
                    }
                )
    return out


def _split_merged_copies(
    holo_atoms: list[dict[str, Any]], expected_n: int
) -> list[list[dict[str, Any]]]:
    """If holo_atoms is an exact multiple of expected_n, split into groups.

    Some PDBs deposit multiple ligand copies as separate residues with
    identical name+seqid (e.g. 1ILH has SRL × 3). gemmi merges them into a
    single residue object. The atom order within the merged residue is
    canonical-CCD-repeated, so we can recover individual copies by
    chunking.
    """
    n = len(holo_atoms)
    if expected_n <= 0 or n % expected_n != 0:
        return [holo_atoms]
    n_copies = n // expected_n
    if n_copies < 2:
        return [holo_atoms]
    return [holo_atoms[i * expected_n : (i + 1) * expected_n] for i in range(n_copies)]


def _build_holo_sdf(
    pdb_id: str,
    ccd: str,
    instance: int,
    holo_atoms: list[dict[str, Any]],
    smiles: str,
    out_path: Path,
) -> tuple[bool, str]:
    """Write a per-holo SDF from heavy-atom holo coords + the CCD SMILES.

    Strategy:
      1. Build a temp PDB block from the holo HETATM entries (element +
         xyz only). RDKit's PDB parser infers bonds from inter-atomic
         distances + valence rules.
      2. Use ``AssignBondOrdersFromTemplate`` against the CCD SMILES to
         restore correct bond orders (single/double/aromatic).
      3. Write the resulting RDKit Mol as an SDF.

    This avoids the brittle atom-order matching against the ideal SDF
    that we hit for PDBs where deposited atoms aren't in canonical CCD
    order. Heavy-atom only — hydrogens are not in X-ray refs anyway.
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")

    if not smiles:
        return False, "no smiles for ccd"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a temp PDB block with the holo heavy atoms (chain X, resname
    # LIG, single residue). RDKit MolFromPDBBlock will infer bonds.
    pdb_lines: list[str] = []
    for i, atom in enumerate(holo_atoms, 1):
        elem = atom["element"]
        # Right-justify atom name for PDB cols 13-16; left-pad short names.
        atom_name = f"{elem}{i}"[:4]
        pdb_lines.append(
            f"HETATM{i:>5} {atom_name:<4} LIG X{1:>4}    "
            f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          "
            f"{elem:>2}"
        )
    pdb_lines.append("END")
    pdb_block = "\n".join(pdb_lines)
    mol_inferred = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
    if mol_inferred is None:
        return False, "rdkit failed to parse temp pdb"
    try:
        Chem.SanitizeMol(mol_inferred)
    except Exception as exc:  # noqa: BLE001
        return False, f"sanitize failed: {exc}"

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return False, f"smiles parse failed: {smiles}"
    template_heavy = Chem.RemoveHs(template)
    if mol_inferred.GetNumAtoms() != template_heavy.GetNumAtoms():
        return (
            False,
            f"heavy-atom count mismatch: holo={mol_inferred.GetNumAtoms()} "
            f"template_smiles={template_heavy.GetNumAtoms()}",
        )
    try:
        mol_final = AllChem.AssignBondOrdersFromTemplate(template_heavy, mol_inferred)
    except Exception as exc:  # noqa: BLE001
        return False, f"AssignBondOrdersFromTemplate failed: {exc}"

    mol_final.SetProp("_Name", f"{pdb_id}_{ccd}_{instance}")
    mol_final.SetProp("pdb_id", pdb_id)
    mol_final.SetProp("ccd_code", ccd)
    mol_final.SetProp("instance", str(instance))
    mol_final.SetProp("smiles", smiles)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol_final)
    writer.close()
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N holo entries (debugging).",
    )
    args = parser.parse_args()

    structure_info = json.loads(STRUCTURE_INFO.read_text())
    ccd_df = pd.read_csv(CCD_CSV)
    ccd_to_smiles = dict(zip(ccd_df["comp_id"], ccd_df["smiles"]))

    print(f"Total PDB entries in {STRUCTURE_INFO.name}: {len(structure_info)}")
    holos = [e for e in structure_info if e["ligands"]]
    print(f"  with at least one ligand: {len(holos)}")
    print(f"CCD codes in {CCD_CSV.name}: {len(ccd_to_smiles)}")

    HOLO_SDF_DIR.mkdir(parents=True, exist_ok=True)
    extracted = _extract_holo_ligands(structure_info)
    if args.limit:
        extracted = extracted[: args.limit]
    print(f"Ligand instances extracted from aligned CIF: {len(extracted)}")

    # Pre-compute heavy-atom counts for each unique CCD so we can split
    # merged multi-copy residues before swapping coords.
    from rdkit import Chem  # noqa: PLC0415

    expected_heavy: dict[str, int] = {}
    for ccd in {e["ccd"] for e in extracted}:
        try:
            sup = Chem.SDMolSupplier(
                str(_download_ideal_sdf(ccd)), removeHs=False, sanitize=True
            )
            mols = [m for m in sup if m is not None]
            if mols:
                expected_heavy[ccd] = Chem.RemoveHs(mols[0]).GetNumAtoms()
        except Exception:  # noqa: BLE001
            pass

    rows: list[dict[str, Any]] = []
    n_ok = 0
    failures: list[str] = []
    for entry in extracted:
        pdb = entry["pdb_id"]
        ccd = entry["ccd"]
        inst = entry["instance"]
        atoms = entry["atoms"]
        # If gemmi merged multiple copies into one residue, split.
        groups = _split_merged_copies(atoms, expected_heavy.get(ccd, 0))
        for sub_idx, group in enumerate(groups):
            real_inst = inst + sub_idx if len(groups) > 1 else inst
            out_sdf = HOLO_SDF_DIR.joinpath(f"{pdb}_{ccd}_{real_inst}.sdf")
            smiles = ccd_to_smiles.get(ccd, "")
            ok, msg = _build_holo_sdf(pdb, ccd, real_inst, group, smiles, out_sdf)
            if ok:
                n_ok += 1
                rows.append(
                    {
                        "pdb_id": pdb,
                        "ccd_code": ccd,
                        "instance": real_inst,
                        "chain_id": entry.get("chain_id", ""),
                        "residue_seqid": entry.get("residue_seqid", 0),
                        "smiles": ccd_to_smiles.get(ccd, ""),
                        "n_heavy_atoms": len(group),
                        "sdf_path": str(out_sdf.relative_to(PROJECT_ROOT)),
                    }
                )
            else:
                failures.append(f"{pdb}/{ccd}/{real_inst}: {msg}")

    print(f"\nBuilt {n_ok} / {len(extracted)} holo ligand SDFs")
    if failures:
        print(f"Failures: {len(failures)}")
        for f in failures[:30]:
            print(f"  {f}")
        if len(failures) > 30:
            print(f"  ... +{len(failures) - 30} more")

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
