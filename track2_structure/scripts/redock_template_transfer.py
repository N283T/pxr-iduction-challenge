#!/usr/bin/env -S pixi run python
"""GLR-style template transfer for Track 2 fragments.

For each Track 2 query SMILES:
  1. Select best holo template from `docs/track2_holo_ligand_db.csv` by MCS
     atom count (whole-molecule Tanimoto is uninformative because our
     queries are fragment-sized while holos are drug-like — substructure
     matching is the right granularity).
  2. Read the holo per-PDB CIF (chain A protein + ligand).
  3. Cα-superpose the holo chain A onto the Boltz model_0 protein for
     the query, apply the same transform to the holo ligand. The
     template ligand now sits in the Boltz reference frame.
  4. RDKit ``AllChem.ConstrainedEmbed`` embeds the query 3D mol with the
     MCS atom subset constrained to the transformed template positions;
     the rest of the query is geometry-optimised around them.
  5. Assemble final PDB: Boltz protein chain A + query ligand chain B
     (residue name LIG, atom serials renumbered, CONECT records remapped),
     in the format the official validator expects.

The output goes to ``structures/boltz2_track2/redock_template/<id>/``
plus a summary CSV at ``docs/track2_redock_template_scores.csv``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

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
HOLO_SDF_DIR = PXR_LBD_DIR.joinpath("holo_ligands_aligned")
TEMPLATE_DB_CSV = PROJECT_ROOT.joinpath("docs", "track2_holo_ligand_db.csv")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")
OUT_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_template")
SCORES_CSV = PROJECT_ROOT.joinpath("docs", "track2_redock_template_scores.csv")


def _per_pdb_cif_path(pdb_id: str) -> Path:
    return PXR_LBD_DIR.joinpath(f"pdb_0000{pdb_id.lower()}_xyz-enrich.cif.gz")


def _read_protein_ca(
    cif_path: Path, chain_id: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Cα xyz, residue numbers) for the longest protein chain."""
    import gemmi  # noqa: PLC0415

    struct = gemmi.read_structure(str(cif_path))
    model = struct[0]
    target_chain = chain_id
    if target_chain is None:
        best = None
        best_n = 0
        for chain in model:
            poly = [r for r in chain if r.het_flag != "H"]
            if len(poly) > best_n:
                best_n = len(poly)
                best = chain.name
        target_chain = best
    ca_xyz: list[tuple[float, float, float]] = []
    resids: list[int] = []
    for chain in model:
        if chain.name != target_chain:
            continue
        for res in chain:
            if res.het_flag == "H":
                continue
            for atom in res:
                if atom.name == "CA":
                    ca_xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    resids.append(res.seqid.num)
    return np.asarray(ca_xyz), np.asarray(resids, dtype=int)


def _read_pdb_ca(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (Cα xyz, residue numbers) from a Boltz model_0 PDB chain A."""
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


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch superposition. Returns (R, t, rmsd) so R@P + t ≈ Q.

    P and Q must be (N, 3) Cα arrays of equal length.
    """
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


def _match_by_residue(
    P_xyz: np.ndarray, P_res: np.ndarray, Q_xyz: np.ndarray, Q_res: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair Cα atoms by residue number between template and Boltz.

    The per-PDB PXR structures use UniProt numbering (residues 142-434
    of O75469) while our Boltz LBD-only model uses 1-293. To match,
    apply the +141 offset to Boltz indices, then intersect on residue
    number.
    """
    LBD_OFFSET = 141
    Q_res_uniprot = Q_res + LBD_OFFSET
    p_lookup = {int(r): i for i, r in enumerate(P_res)}
    q_lookup = {int(r): i for i, r in enumerate(Q_res_uniprot)}
    common = sorted(set(p_lookup) & set(q_lookup))
    if len(common) < 50:
        # Fall back to leading-N truncation when residue numbering does
        # not match (some entries may use different conventions).
        n = min(len(P_xyz), len(Q_xyz))
        return P_xyz[:n], Q_xyz[:n]
    P_sel = np.stack([P_xyz[p_lookup[r]] for r in common])
    Q_sel = np.stack([Q_xyz[q_lookup[r]] for r in common])
    return P_sel, Q_sel


def _select_best_template(
    query_smiles: str, templates_df: pd.DataFrame, min_mcs: int = 6
) -> dict[str, Any] | None:
    """Pick the template with the largest MCS atom count.

    Returns None when no template exceeds ``min_mcs`` heavy atoms.
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFMCS  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    qm = Chem.MolFromSmiles(query_smiles)
    if qm is None:
        return None
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
        if mcs.numAtoms < min_mcs:
            continue
        if best is None or mcs.numAtoms > best["mcs_atoms"]:
            best = {
                "ccd": row.ccd_code,
                "pdb": row.pdb_id,
                "instance": int(row.instance),
                "smiles": row.smiles,
                "sdf_path": str(PROJECT_ROOT.joinpath(row.sdf_path)),
                "mcs_atoms": mcs.numAtoms,
                "mcs_smarts": mcs.smartsString,
            }
    return best


def _transform_sdf_coords(sdf_path: Path, R: np.ndarray, t: np.ndarray) -> "Chem.Mol":  # type: ignore  # noqa: F821
    """Load template SDF and apply (R, t) so coords land in the Boltz frame."""
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    sup = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    mols = [m for m in sup if m is not None]
    if not mols:
        raise RuntimeError(f"could not load template sdf {sdf_path}")
    mol = mols[0]
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = np.array([p.x, p.y, p.z])
        v2 = R @ v + t
        conf.SetAtomPosition(i, Point3D(float(v2[0]), float(v2[1]), float(v2[2])))
    return mol


def _constrained_embed_query(
    query_smiles: str, template_mol, mcs_smarts: str
) -> "Chem.Mol":  # type: ignore  # noqa: F821
    """Embed query 3D so the MCS atom subset overlays the template positions."""
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    qm = Chem.AddHs(Chem.MolFromSmiles(query_smiles))
    if qm is None:
        raise RuntimeError(f"smiles parse failed: {query_smiles}")
    pattern = Chem.MolFromSmarts(mcs_smarts)
    if pattern is None:
        raise RuntimeError("MCS smarts did not parse")
    q_match = qm.GetSubstructMatch(pattern)
    t_match = template_mol.GetSubstructMatch(pattern)
    if not q_match or not t_match or len(q_match) != len(t_match):
        raise RuntimeError(
            f"MCS atom mapping mismatch: q={len(q_match)} t={len(t_match)}"
        )
    # ConstrainedEmbed wants a "core" mol whose conformer atoms match the
    # query subset by index. Build it explicitly: a clone of the template
    # restricted to the matched atoms in q_match order.
    core = Chem.RWMol()
    pos_map = {}
    for new_idx, t_idx in enumerate(t_match):
        old_atom = template_mol.GetAtomWithIdx(t_idx)
        new_atom = Chem.Atom(old_atom.GetAtomicNum())
        new_atom.SetFormalCharge(old_atom.GetFormalCharge())
        core.AddAtom(new_atom)
        pos_map[t_idx] = new_idx
    # Copy bonds inside the matched subset
    for bond in template_mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in pos_map and b in pos_map:
            core.AddBond(pos_map[a], pos_map[b], bond.GetBondType())
    core_mol = core.GetMol()
    Chem.SanitizeMol(core_mol)
    # Set core conformer positions from template_mol coords
    conf = Chem.Conformer(core_mol.GetNumAtoms())
    template_conf = template_mol.GetConformer()
    for old_idx, new_idx in pos_map.items():
        p = template_conf.GetAtomPosition(old_idx)
        conf.SetAtomPosition(new_idx, p)
    core_mol.AddConformer(conf, assignId=True)

    # Build the q_match → core index map for ConstrainedEmbed
    coord_map = {
        q_idx: core_mol.GetConformer().GetAtomPosition(pos_map[t_idx])
        for q_idx, t_idx in zip(q_match, t_match)
    }
    try:
        AllChem.ConstrainedEmbed(qm, core_mol, useTethers=True)
    except Exception as exc:  # noqa: BLE001
        # Fallback: manual coord-map embedding
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.coordMap = coord_map
        cid = AllChem.EmbedMolecule(qm, params)
        if cid < 0:
            raise RuntimeError(f"constrained embed failed: {exc}")
        AllChem.MMFFOptimizeMolecule(qm, maxIters=200)
    return Chem.RemoveHs(qm)


def _assemble_pdb(boltz_pdb: Path, refined_query_mol, out_pdb: Path) -> None:
    """Combine Boltz protein chain A + new ligand with proper LIG residue + CONECT."""
    from rdkit import Chem  # noqa: PLC0415

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein_text = boltz_pdb.read_text()
    # Strip the existing Boltz ligand (HETATM resname LIG and any CONECT
    # lines that reference its serials). We replace it with the
    # template-transferred query ligand below.
    boltz_lig_serials: set[int] = set()
    for ln in protein_text.splitlines():
        if ln.startswith("HETATM") and ln[17:20].strip() == "LIG":
            try:
                boltz_lig_serials.add(int(ln[6:11].strip()))
            except (ValueError, IndexError):
                pass

    def _conect_references_lig(line: str) -> bool:
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
        if ln.startswith("CONECT") and _conect_references_lig(ln):
            continue
        protein_lines.append(ln)

    last_serial = 0
    for line in protein_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                last_serial = max(last_serial, int(line[6:11].strip()))
            except (ValueError, IndexError):
                pass

    refined_pdb_block = Chem.MolToPDBBlock(refined_query_mol)
    serial_remap: dict[int, int] = {}
    next_serial = last_serial + 1
    lig_atom_lines: list[str] = []
    lig_conect_lines: list[str] = []
    for line in refined_pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                old_serial = int(line[6:11].strip())
            except ValueError:
                continue
            new_serial = next_serial
            next_serial += 1
            serial_remap[old_serial] = new_serial
            atom_name = line[12:16]
            if len(line) < 78:
                line = line.ljust(78)
            element = line[76:78]
            x = line[30:38]
            y = line[38:46]
            z = line[46:54]
            occ = line[54:60] if len(line) >= 60 else "  1.00"
            bfac = line[60:66] if len(line) >= 66 else "  0.00"
            lig_atom_lines.append(
                f"HETATM{new_serial:>5} {atom_name:<4} LIG B   1    "
                f"{x}{y}{z}{occ}{bfac}          {element}"
            )
        elif line.startswith("CONECT"):
            rest = line[6:]
            old_atoms: list[int] = []
            for i in range(0, len(rest), 5):
                cell = rest[i : i + 5].strip()
                if not cell:
                    continue
                try:
                    old_atoms.append(int(cell))
                except ValueError:
                    pass
            new_atoms = [serial_remap[a] for a in old_atoms if a in serial_remap]
            if len(new_atoms) >= 2:
                lig_conect_lines.append(
                    "CONECT" + "".join(f"{a:>5}" for a in new_atoms)
                )

    if not lig_atom_lines:
        raise RuntimeError("no ligand atoms produced")

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


def _process_one(qid: str, qsmi: str, templates_df: pd.DataFrame) -> dict[str, Any]:
    base = {"compound": qid, "smiles": qsmi}
    boltz_pdb = PRED_DIR.joinpath(qid, f"{qid}_model_0.pdb")
    if not boltz_pdb.exists():
        return {**base, "error": "boltz_pdb_missing"}

    best = _select_best_template(qsmi, templates_df)
    if best is None:
        return {**base, "error": "no_template_with_min_mcs"}

    base.update(
        {
            "template_pdb": best["pdb"],
            "template_ccd": best["ccd"],
            "template_smiles": best["smiles"],
            "mcs_atoms": best["mcs_atoms"],
        }
    )

    cif_path = _per_pdb_cif_path(best["pdb"])
    if not cif_path.exists():
        return {**base, "error": f"per_pdb_cif_missing: {cif_path.name}"}
    template_ca, template_res = _read_protein_ca(cif_path)
    boltz_ca, boltz_res = _read_pdb_ca(boltz_pdb)
    if len(template_ca) == 0 or len(boltz_ca) == 0:
        return {**base, "error": "empty_ca"}
    P, Q = _match_by_residue(template_ca, template_res, boltz_ca, boltz_res)
    R, t, ca_rmsd = _kabsch(P, Q)
    base["n_matched_ca"] = int(len(P))
    base["ca_rmsd"] = ca_rmsd

    try:
        template_mol = _transform_sdf_coords(Path(best["sdf_path"]), R, t)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"sdf_transform_failed: {exc}"}

    try:
        new_query = _constrained_embed_query(qsmi, template_mol, best["mcs_smarts"])
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"embed_failed: {exc}"}

    out_pdb = OUT_DIR.joinpath(qid, f"{qid}_template.pdb")
    try:
        _assemble_pdb(boltz_pdb, new_query, out_pdb)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"assemble_failed: {exc}"}

    base["out_pdb"] = str(out_pdb.relative_to(PROJECT_ROOT))
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--templates-csv",
        type=Path,
        default=TEMPLATE_DB_CSV,
        help="Holo template database (output of build_holo_ligand_db.py).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    templates_df = (
        pd.read_csv(args.templates_csv).drop_duplicates("smiles").reset_index(drop=True)
    )
    print(f"Queries: {len(df)} | Templates (unique smiles): {len(templates_df)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n, (_, qrow) in enumerate(df.iterrows(), 1):
        r = _process_one(qrow.structure, qrow.smiles, templates_df)
        rows.append(r)
        if n % 20 == 0 or n == len(df):
            errs = sum(1 for x in rows if "error" in x)
            print(f"  {n}/{len(df)}  errors={errs}")

    out_df = pd.DataFrame(rows).sort_values("compound").reset_index(drop=True)
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(SCORES_CSV, index=False)
    print(f"\nWrote {SCORES_CSV}")
    if "error" in out_df.columns:
        errs = out_df[out_df["error"].notna()]
        print(f"Errors: {len(errs)}")
        for _, r in errs.head(20).iterrows():
            print(f"  {r['compound']}: {r['error']}")
    ok = out_df[out_df["error"].isna()] if "error" in out_df.columns else out_df
    if not ok.empty:
        print(f"Succeeded: {len(ok)} / {len(df)}")
        if "ca_rmsd" in ok.columns and ok["ca_rmsd"].notna().any():
            print(
                f"  Cα RMSD (template→Boltz): mean={ok['ca_rmsd'].mean():.2f}Å "
                f"median={ok['ca_rmsd'].median():.2f}Å max={ok['ca_rmsd'].max():.2f}Å"
            )
        if "mcs_atoms" in ok.columns:
            print(
                f"  MCS atoms used: mean={ok['mcs_atoms'].mean():.1f} "
                f"median={ok['mcs_atoms'].median():.1f}"
            )


if __name__ == "__main__":
    main()
