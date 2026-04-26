#!/usr/bin/env -S pixi run python
"""OpenEye POSIT pose prediction for Track 2 fragments.

Uses POSIT (HYBRID + MCS + SHAPEFIT + FRED ensemble) to dock each query
against the protein + bound-ligand of the best matching holo template.
Pose is then Cα-transformed back into our Boltz model_0 reference frame
so the assembled PDB drops into the same submission format as the other
v* zips.

Per-query workflow:
  1. Look up best holo template from
     ``docs/track2_redock_template_scores.csv`` (already computed by
     ``redock_template_transfer.py``).
  2. Prepare an OE receptor for that template's per-PDB CIF (chain that
     owns the selected ligand instance). Cached across queries that
     share a template.
  3. Generate up to 50 query conformers via OEOmega.
  4. ``OEPosit.Dock`` against the receptor. POSIT auto-picks among
     SHAPEFIT / HYBRID / MCS / FRED based on the query.
  5. Cα-superpose template per-PDB protein onto Boltz model_0 (per-query
     residue-number match) → apply transform to POSIT pose.
  6. Strip the Boltz LIG residue and graft the new POSIT-derived ligand
     into Boltz model_0's PDB (proper LIG resname, chain B, renumbered
     serials, remapped CONECT records).

Failures (POSIT no-pose / receptor build failure / template missing)
fall back to Boltz model_0 so the submission stays at 184 entries.

Usage:
    pixi run python track2_structure/scripts/redock_posit.py
    pixi run python track2_structure/scripts/redock_posit.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("OE_LICENSE", os.path.expanduser("~/.openeye/oe_license.txt"))

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from track2.constants import REPO_ROOT as PROJECT_ROOT  # noqa: E402

PRED_DIR = PROJECT_ROOT.joinpath(
    "structures",
    "boltz2_track2",
    "outputs",
    "holo",
    "boltz_results_holo",
    "predictions",
)
PXR_LBD_DIR = PROJECT_ROOT.joinpath("structures", "pxr_lbd")
TEMPLATE_DB_CSV = PROJECT_ROOT.joinpath("docs", "track2_holo_ligand_db.csv")
TEMPLATE_SCORES_CSV = PROJECT_ROOT.joinpath("docs", "track2_redock_template_scores.csv")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")
OUT_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_posit")
SCORES_CSV = PROJECT_ROOT.joinpath("docs", "track2_redock_posit_scores.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cp = P.mean(0)
    cq = Q.mean(0)
    Pc = P - cp
    Qc = Q - cq
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    aligned = (R @ Pc.T).T + cq
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean()))
    return R, t, rmsd


def _per_pdb_cif(pdb_id: str) -> Path:
    return PXR_LBD_DIR.joinpath(f"pdb_0000{pdb_id.lower()}_xyz-enrich.cif.gz")


def _read_pdb_ca(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[tuple[float, float, float]] = []
    resids: list[int] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            r = int(line[22:26].strip())
        except ValueError:
            continue
        xyz.append((x, y, z))
        resids.append(r)
    return np.asarray(xyz), np.asarray(resids, dtype=int)


def _read_protein_ca_cif(
    cif_path: Path, chain_id: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    import gemmi  # noqa: PLC0415

    struct = gemmi.read_structure(str(cif_path))
    model = struct[0]
    target = chain_id
    if target is None:
        best_chain = None
        best_n = 0
        for chain in model:
            poly = [r for r in chain if r.het_flag != "H"]
            if len(poly) > best_n:
                best_n = len(poly)
                best_chain = chain.name
        target = best_chain
    xyz: list[tuple[float, float, float]] = []
    resids: list[int] = []
    for chain in model:
        if chain.name != target:
            continue
        for res in chain:
            if res.het_flag == "H":
                continue
            for atom in res:
                if atom.name == "CA":
                    xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    resids.append(res.seqid.num)
    return np.asarray(xyz), np.asarray(resids, dtype=int)


def _match_resids(
    P_xyz: np.ndarray, P_res: np.ndarray, Q_xyz: np.ndarray, Q_res: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    LBD_OFFSET = 141
    Q_res_uniprot = Q_res + LBD_OFFSET
    p_lookup = {int(r): i for i, r in enumerate(P_res)}
    q_lookup = {int(r): i for i, r in enumerate(Q_res_uniprot)}
    common = sorted(set(p_lookup) & set(q_lookup))
    if len(common) < 50:
        n = min(len(P_xyz), len(Q_xyz))
        return P_xyz[:n], Q_xyz[:n]
    P_sel = np.stack([P_xyz[p_lookup[r]] for r in common])
    Q_sel = np.stack([Q_xyz[q_lookup[r]] for r in common])
    return P_sel, Q_sel


# ---------------------------------------------------------------------------
# OE protein/ligand split + receptor preparation (cached per (pdb, chain, ccd))
# ---------------------------------------------------------------------------


def _build_pdb_blocks(cif_path: Path, chain_id: str, ccd: str) -> tuple[str, str]:
    """Return (protein_pdb_block, ligand_pdb_block) for the given chain.

    Only includes the ONE specified chain (homodimer-aware) and the LIGAND
    residues whose CCD matches ``ccd`` within that chain.
    """
    import gemmi  # noqa: PLC0415

    struct = gemmi.read_structure(str(cif_path))
    prot_lines: list[str] = []
    lig_lines: list[str] = []
    serial = 0
    for chain in struct[0]:
        if chain.name != chain_id:
            continue
        for res in chain:
            for atom in res:
                serial += 1
                rec = "HETATM" if res.het_flag == "H" else "ATOM  "
                line = (
                    f"{rec}{serial:>5} {atom.name:<4} {res.name:<3} "
                    f"{chain.name:1s}{res.seqid.num:>4}    "
                    f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                    f"  1.00  0.00          {atom.element.name:>2}"
                )
                if res.name == ccd and res.het_flag == "H":
                    lig_lines.append(line)
                elif res.het_flag != "H":
                    prot_lines.append(line)
    prot_block = "\n".join(prot_lines) + "\nEND\n"
    lig_block = "\n".join(lig_lines) + "\nEND\n"
    return prot_block, lig_block


def _load_oe_mol_from_pdb(pdb_block: str):
    from openeye import oechem  # noqa: PLC0415

    mol = oechem.OEGraphMol()
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_PDB)
    ifs.openstring(pdb_block)
    oechem.OEReadMolecule(ifs, mol)
    ifs.close()
    return mol


def _build_receptor_cached(pdb: str, chain_id: str, ccd: str, cache: dict):
    """Build (or fetch from cache) an OE receptor for the (pdb, chain, ccd) tuple."""
    from openeye import oechem, oedocking  # noqa: PLC0415

    key = (pdb, chain_id, ccd)
    if key in cache:
        return cache[key]
    cif_path = _per_pdb_cif(pdb)
    if not cif_path.exists():
        cache[key] = None
        return None
    prot_block, lig_block = _build_pdb_blocks(cif_path, chain_id, ccd)
    if not lig_block.strip().split("\n")[0]:
        cache[key] = None
        return None
    prot_mol = _load_oe_mol_from_pdb(prot_block)
    lig_mol = _load_oe_mol_from_pdb(lig_block)
    receptor = oechem.OEGraphMol()
    if not oedocking.OEMakeReceptor(receptor, prot_mol, lig_mol):
        cache[key] = None
        return None
    if not oedocking.OEIsReceptor(receptor):
        cache[key] = None
        return None
    cache[key] = receptor
    return receptor


# ---------------------------------------------------------------------------
# POSIT docking + transform back to Boltz frame
# ---------------------------------------------------------------------------


def _omega_query(smiles: str, max_confs: int = 50):
    from openeye import oechem, oeomega  # noqa: PLC0415

    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        return None
    om = oeomega.OEOmega()
    om.SetMaxConfs(max_confs)
    om.SetStrictStereo(False)
    if not om(mol):
        return None
    return mol


def _posit_dock(query_oe, receptor):
    from openeye import oedocking  # noqa: PLC0415

    posit = oedocking.OEPosit(oedocking.OEPositOptions())
    if not posit.AddReceptor(receptor):
        return None
    result = oedocking.OEPositResults()
    rc = posit.Dock(result, query_oe)
    if rc != 0 and result.GetNumPoses() == 0:
        return None
    poses = list(result.GetSinglePoseResults())
    if not poses:
        return None
    return poses[0]


def _oe_pose_to_xyz(pose_mol) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    """Extract (element list, xyz array, bond list) from an OE pose.

    Bond list is (atom_idx_a, atom_idx_b, bond_order) tuples — needed for
    CONECT records when assembling the final PDB.
    """
    from openeye import oechem  # noqa: PLC0415

    conf = pose_mol.GetActive() if hasattr(pose_mol, "GetActive") else pose_mol
    elements: list[str] = []
    xyz_list: list[tuple[float, float, float]] = []
    idx_map: dict[int, int] = {}
    # Heavy atoms only — the official validator parses ligand PDBs with
    # ``removeHs=True`` against a heavy-atom-only template, so any
    # explicit hydrogens left over from OEOmega break
    # AssignBondOrdersFromTemplate.
    for atom in conf.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        elements.append(oechem.OEGetAtomicSymbol(atom.GetAtomicNum()))
        c = conf.GetCoords(atom)
        xyz_list.append((c[0], c[1], c[2]))
        idx_map[atom.GetIdx()] = len(elements) - 1
    bonds: list[tuple[int, int, int]] = []
    for bond in conf.GetBonds():
        a = bond.GetBgnIdx()
        b = bond.GetEndIdx()
        if a in idx_map and b in idx_map:
            bonds.append((idx_map[a], idx_map[b], bond.GetOrder()))
    return elements, np.asarray(xyz_list), bonds


def _assemble_pdb(
    boltz_pdb: Path,
    elements: list[str],
    pose_xyz: np.ndarray,
    bonds: list[tuple[int, int, int]],
    out_pdb: Path,
) -> None:
    """Boltz protein chain A + new POSIT-derived ligand → submission PDB."""
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein_text = boltz_pdb.read_text()
    boltz_lig_serials: set[int] = set()
    for ln in protein_text.splitlines():
        if ln.startswith("HETATM") and ln[17:20].strip() == "LIG":
            try:
                boltz_lig_serials.add(int(ln[6:11].strip()))
            except (ValueError, IndexError):
                pass

    def conect_refers_lig(line: str) -> bool:
        rest = line[6:]
        for i in range(0, len(rest), 5):
            cell = rest[i : i + 5].strip()
            if cell and int(cell) in boltz_lig_serials:
                return True
        return False

    protein_lines: list[str] = []
    for ln in protein_text.splitlines():
        if ln.strip() == "END":
            continue
        if ln.startswith("HETATM") and ln[17:20].strip() == "LIG":
            continue
        if ln.startswith("CONECT") and conect_refers_lig(ln):
            continue
        protein_lines.append(ln)

    last_serial = 0
    for line in protein_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                last_serial = max(last_serial, int(line[6:11].strip()))
            except (ValueError, IndexError):
                pass

    next_serial = last_serial + 1
    serial_remap: dict[int, int] = {}
    lig_atom_lines: list[str] = []
    for i, (elem, (x, y, z)) in enumerate(zip(elements, pose_xyz)):
        atom_name = f"{elem}{i + 1}"[:4]
        new_serial = next_serial
        next_serial += 1
        serial_remap[i] = new_serial
        lig_atom_lines.append(
            f"HETATM{new_serial:>5} {atom_name:<4} LIG B   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}"
        )

    lig_conect_lines: list[str] = []
    # Group bonds per atom to emit one CONECT per source atom
    from collections import defaultdict  # noqa: PLC0415

    by_a: dict[int, list[int]] = defaultdict(list)
    for a, b, order in bonds:
        sa = serial_remap[a]
        sb = serial_remap[b]
        # Repeat the partner per bond order to match PDB CONECT convention
        for _ in range(max(1, order)):
            by_a[sa].append(sb)
            by_a[sb].append(sa)
    for sa, partners in by_a.items():
        line = "CONECT" + f"{sa:>5}" + "".join(f"{p:>5}" for p in partners)
        lig_conect_lines.append(line)

    last_atom_idx = max(
        (i for i, ln in enumerate(protein_lines) if ln.startswith("ATOM")),
        default=-1,
    )
    if last_atom_idx >= 0 and (
        last_atom_idx + 1 >= len(protein_lines)
        or not protein_lines[last_atom_idx + 1].startswith("TER")
    ):
        protein_lines.insert(last_atom_idx + 1, "TER")

    out_pdb.write_text(
        "\n".join(protein_lines)
        + "\n"
        + "\n".join(lig_atom_lines)
        + "\n"
        + "\n".join(lig_conect_lines)
        + ("\n" if lig_conect_lines else "")
        + "END\n"
    )


def _process_one(
    qid: str, qsmi: str, templates_df: pd.DataFrame, recept_cache: dict
) -> dict[str, Any]:
    base = {"compound": qid}
    boltz_pdb = PRED_DIR.joinpath(qid, f"{qid}_model_0.pdb")
    if not boltz_pdb.exists():
        return {**base, "error": "boltz_pdb_missing"}

    # Find best holo template by MCS (re-using build script's selection)
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFMCS  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    qm = Chem.MolFromSmiles(qsmi)
    if qm is None:
        return {**base, "error": "smiles_parse"}
    best = None
    for _, row in templates_df.iterrows():
        tm = Chem.MolFromSmiles(row.smiles)
        if tm is None:
            continue
        try:
            mcs = rdFMCS.FindMCS(
                [qm, tm],
                timeout=2,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            )
        except Exception:  # noqa: BLE001
            continue
        if mcs.numAtoms < 6:
            continue
        if best is None or mcs.numAtoms > best["mcs_atoms"]:
            best = {
                "ccd": row.ccd_code,
                "pdb": row.pdb_id,
                "chain_id": row.get("chain_id", None),
                "smiles": row.smiles,
                "mcs_atoms": mcs.numAtoms,
            }
    if best is None:
        return {**base, "error": "no_template_with_min_mcs"}
    base.update(
        {
            "template_pdb": best["pdb"],
            "template_ccd": best["ccd"],
            "template_chain": best.get("chain_id"),
            "mcs_atoms": best["mcs_atoms"],
        }
    )

    # Build receptor (cached)
    receptor = _build_receptor_cached(
        best["pdb"], best.get("chain_id") or "A", best["ccd"], recept_cache
    )
    if receptor is None:
        return {**base, "error": "receptor_build_failed"}

    # Conformer + dock
    query_oe = _omega_query(qsmi, max_confs=50)
    if query_oe is None:
        return {**base, "error": "omega_failed"}
    pose = _posit_dock(query_oe, receptor)
    if pose is None:
        return {**base, "error": "posit_no_pose"}
    base["posit_probability"] = float(pose.GetProbability())
    from openeye import oedocking  # noqa: PLC0415

    base["posit_method"] = oedocking.OEPositMethodGetName(pose.GetPositMethod())

    # Cα-superpose template per-PDB protein onto Boltz model_0 protein,
    # then apply (R, t) to the POSIT pose so it lands in our submission frame.
    cif_path = _per_pdb_cif(best["pdb"])
    template_ca, template_res = _read_protein_ca_cif(cif_path, best.get("chain_id"))
    boltz_ca, boltz_res = _read_pdb_ca(boltz_pdb)
    P, Q = _match_resids(template_ca, template_res, boltz_ca, boltz_res)
    R, t, ca_rmsd = _kabsch(P, Q)
    base["ca_rmsd"] = float(ca_rmsd)

    pose_mol = pose.GetPose()
    elements, xyz, bonds = _oe_pose_to_xyz(pose_mol)
    xyz_boltz = (R @ xyz.T).T + t

    out_pdb = OUT_DIR.joinpath(qid, f"{qid}_posit.pdb")
    try:
        _assemble_pdb(boltz_pdb, elements, xyz_boltz, bonds, out_pdb)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"assemble_failed: {exc}"}
    base["out_pdb"] = str(out_pdb.relative_to(PROJECT_ROOT))
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--templates-csv", type=Path, default=TEMPLATE_DB_CSV)
    parser.add_argument("--out-csv", type=Path, default=SCORES_CSV)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    templates_df = (
        pd.read_csv(args.templates_csv).drop_duplicates("smiles").reset_index(drop=True)
    )
    print(f"Queries: {len(df)} | Unique-smiles templates: {len(templates_df)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recept_cache: dict = {}
    rows: list[dict[str, Any]] = []
    for n, (_, row) in enumerate(df.iterrows(), 1):
        rows.append(_process_one(row.structure, row.smiles, templates_df, recept_cache))
        if n % 10 == 0 or n == len(df):
            errs = sum(1 for r in rows if "error" in r)
            print(f"  {n}/{len(df)}  errors={errs}")

    out_df = pd.DataFrame(rows).sort_values("compound").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    if "error" in out_df.columns:
        errs = out_df[out_df["error"].notna()]
        print(f"Errors: {len(errs)}")
        for _, r in errs.head(20).iterrows():
            print(f"  {r['compound']}: {r['error']}")
    ok = out_df[out_df.get("error", pd.Series([None] * len(out_df))).isna()]
    if not ok.empty:
        print(f"\nSucceeded: {len(ok)}")
        if "posit_probability" in ok.columns:
            print(
                f"  POSIT probability: mean={ok['posit_probability'].mean():.3f} "
                f"median={ok['posit_probability'].median():.3f}"
            )
        if "posit_method" in ok.columns:
            print("  Method distribution:")
            for m, c in ok["posit_method"].value_counts().items():
                print(f"    {m}: {c}")
        if "ca_rmsd" in ok.columns:
            print(
                f"  Cα RMSD (template→Boltz): mean={ok['ca_rmsd'].mean():.2f}Å "
                f"median={ok['ca_rmsd'].median():.2f}Å"
            )


if __name__ == "__main__":
    main()
