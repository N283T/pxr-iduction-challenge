#!/usr/bin/env -S pixi run python
"""Self-evaluate Track 2 pose candidates against the best holo template.

Idea: until the LB scores our submissions we have no ground truth. But for
queries that share a substantive (MCS >= 8 heavy atoms) substructure with
some holo crystal, the holo's binding mode is a *prior* on where the
shared scaffold should sit. By computing the **MCS-atom RMSD** between
each candidate pose and the holo template (after putting both in a
common protein frame), we get a non-circular, experimentally-grounded
self-evaluation signal:

  - Across Boltz model_0..4, gnina-refined, medoid, and the
    template-transferred pose, the candidate with the smallest
    template-RMSD is the most holo-consistent on its shared scaffold.
  - For queries with MCS < 8, the holo prior is too weak to be
    informative — we skip them and accept Boltz model_0 as the default.

Coordinate frame handling: each Boltz `model_i.pdb` is output in its own
diffusion-sample frame. Before computing distances against the template
(which we Cα-superpose to `model_0`), we also Cα-superpose `model_1..4`
to `model_0`. gnina-refined and template-transferred poses are already
based on `model_0`'s protein, so they need no realignment.

Output:
  docs/track2_template_rmsd_scores.csv with one row per (query, source)
  pair, plus a per-query "best source" derived column.
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
GNINA_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_gnina")
TEMPLATE_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_template")
HOLO_LIGAND_DIR = PROJECT_ROOT.joinpath("structures", "pxr_lbd", "holo_ligands_aligned")
TEMPLATE_DB_CSV = PROJECT_ROOT.joinpath("docs", "track2_holo_ligand_db.csv")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")
OUT_CSV = PROJECT_ROOT.joinpath("docs", "track2_template_rmsd_scores.csv")


# ---------------------------------------------------------------------------
# Geometry helpers
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
    """Read Cα + residue numbers from a per-PDB CIF.

    For homodimers (e.g. 5X0R has chains A and B both with ligand
    instances) we MUST use the chain that owns the selected template
    ligand. Otherwise the Cα superposition is between *protein from chain
    B* and our Boltz output, while the *ligand from chain A* gets
    transformed using that wrong (R, t), landing it 20+ Å away from the
    pocket. Caller passes chain_id; when None we fall back to the longest
    polymer chain.
    """
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
# Ligand extraction
# ---------------------------------------------------------------------------


def _extract_ligand_mol(pdb_path: Path, smiles: str):
    """Read ligand from PDB, AssignBondOrdersFromTemplate(SMILES) so atom
    indices match the SMILES canonical Mol. Returns RDKit Mol with conformer.
    """
    import tempfile  # noqa: PLC0415
    import MDAnalysis as mda  # noqa: PLC0415
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    u = mda.Universe(str(pdb_path))
    lig = u.select_atoms("resname LIG")
    if len(lig) == 0:
        raise RuntimeError("no LIG residue")
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        lig.write(tmp.name)
        lig_pdb = Path(tmp.name)
    try:
        ref = Chem.MolFromSmiles(smiles)
        if ref is None:
            raise RuntimeError("smiles parse failed")
        m = Chem.MolFromPDBFile(str(lig_pdb), removeHs=True, sanitize=False)
        if m is None:
            raise RuntimeError("pdb ligand parse failed")
        m = AllChem.AssignBondOrdersFromTemplate(ref, m)
    finally:
        lig_pdb.unlink(missing_ok=True)
    return m


def _mcs_rmsd(candidate_mol, template_mol, mcs_smarts: str) -> float:
    """Atom-paired RMSD on the MCS subset, minimised over all symmetric
    MCS atom mappings.

    A symmetric scaffold (phenyl ring, sulfonamide, etc.) has multiple
    valid GetSubstructMatch orderings. Without iterating over all
    template matches we end up pairing physically distinct atoms which
    can produce 40+ Å RMSDs. Iterating over both ``GetSubstructMatches``
    is O(n_match^2) but n_match is small in practice.
    """
    from rdkit import Chem  # noqa: PLC0415

    pattern = Chem.MolFromSmarts(mcs_smarts)
    if pattern is None:
        raise RuntimeError("MCS smarts parse failed")
    c_matches = candidate_mol.GetSubstructMatches(pattern, uniquify=False)
    t_matches = template_mol.GetSubstructMatches(pattern, uniquify=False)
    if not c_matches or not t_matches:
        raise RuntimeError("mcs has no match in candidate or template")

    c_conf = candidate_mol.GetConformer()
    t_conf = template_mol.GetConformer()

    def pair_rmsd(c_idx, t_idx):
        if len(c_idx) != len(t_idx):
            return float("inf")
        diffs = []
        for ci, ti in zip(c_idx, t_idx):
            cp = c_conf.GetAtomPosition(ci)
            tp = t_conf.GetAtomPosition(ti)
            diffs.append((cp.x - tp.x) ** 2 + (cp.y - tp.y) ** 2 + (cp.z - tp.z) ** 2)
        return float(np.sqrt(np.mean(diffs)))

    best = float("inf")
    for c_idx in c_matches:
        for t_idx in t_matches:
            r = pair_rmsd(c_idx, t_idx)
            if r < best:
                best = r
    if not np.isfinite(best):
        raise RuntimeError("no valid mcs pairing")
    return best


def _per_pdb_cif(pdb_id: str) -> Path:
    return PROJECT_ROOT.joinpath(
        "structures",
        "pxr_lbd",
        f"pdb_0000{pdb_id.lower()}_xyz-enrich.cif.gz",
    )


def _transform_template_mol(sdf_path: Path, R: np.ndarray, t: np.ndarray):
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    sup = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    mols = [m for m in sup if m is not None]
    if not mols:
        raise RuntimeError("template sdf parse failed")
    mol = mols[0]
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = R @ np.array([p.x, p.y, p.z]) + t
        conf.SetAtomPosition(i, Point3D(*v.tolist()))
    return mol


def _superpose_pdb_to_ref(
    cand_pdb: Path, ref_ca: np.ndarray, ref_res: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (R, t) that maps the candidate PDB's protein onto the ref."""
    cand_ca, cand_res = _read_pdb_ca(cand_pdb)
    LBD_OFFSET = 0  # both use same LBD numbering for Boltz outputs
    p_lookup = {int(r): i for i, r in enumerate(cand_res)}
    q_lookup = {int(r + LBD_OFFSET): i for i, r in enumerate(ref_res)}
    common = sorted(set(p_lookup) & set(q_lookup))
    if len(common) < 50:
        return np.eye(3), np.zeros(3)
    P = np.stack([cand_ca[p_lookup[r]] for r in common])
    Q = np.stack([ref_ca[q_lookup[r]] for r in common])
    R, t, _rmsd = _kabsch(P, Q)
    return R, t


def _transform_ligand_in_pdb(pdb_path: Path, smiles: str, R: np.ndarray, t: np.ndarray):
    """Extract ligand mol from a PDB and apply (R, t) so its coords are
    in the reference (= model_0) frame.
    """
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    mol = _extract_ligand_mol(pdb_path, smiles)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = R @ np.array([p.x, p.y, p.z]) + t
        conf.SetAtomPosition(i, Point3D(*v.tolist()))
    return mol


# ---------------------------------------------------------------------------
# Per-query scoring
# ---------------------------------------------------------------------------


def _select_template(qsmi: str, templates_df: pd.DataFrame, min_mcs: int = 8):
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFMCS  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    qm = Chem.MolFromSmiles(qsmi)
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
                "chain_id": row.get("chain_id", None),
                "smiles": row.smiles,
                "sdf_path": str(PROJECT_ROOT.joinpath(row.sdf_path)),
                "mcs_atoms": mcs.numAtoms,
                "mcs_smarts": mcs.smartsString,
            }
    return best


def _process_one(qid: str, qsmi: str, templates_df: pd.DataFrame, min_mcs: int):
    boltz_dir = PRED_DIR.joinpath(qid)
    model0 = boltz_dir.joinpath(f"{qid}_model_0.pdb")
    if not model0.exists():
        return [{"compound": qid, "error": "boltz_model_0_missing"}]
    qha = 0
    try:
        from rdkit import Chem  # noqa: PLC0415

        qm = Chem.MolFromSmiles(qsmi)
        if qm is not None:
            qha = qm.GetNumHeavyAtoms()
    except Exception:  # noqa: BLE001
        pass

    best = _select_template(qsmi, templates_df, min_mcs=min_mcs)
    if best is None:
        return [
            {
                "compound": qid,
                "qha": qha,
                "skipped": "mcs_below_threshold",
            }
        ]

    # Cα-superpose template's per-PDB protein onto model_0
    cif_path = _per_pdb_cif(best["pdb"])
    if not cif_path.exists():
        return [{"compound": qid, "error": f"per_pdb_cif_missing:{best['pdb']}"}]
    template_ca, template_res = _read_protein_ca_cif(cif_path, best.get("chain_id"))
    boltz_ca, boltz_res = _read_pdb_ca(model0)
    P, Q = _match_resids(template_ca, template_res, boltz_ca, boltz_res)
    R_t, t_t, _ = _kabsch(P, Q)
    try:
        template_mol = _transform_template_mol(Path(best["sdf_path"]), R_t, t_t)
    except Exception as exc:  # noqa: BLE001
        return [{"compound": qid, "error": f"template_transform_failed:{exc}"}]

    rows = []
    base = {
        "compound": qid,
        "qha": qha,
        "template_pdb": best["pdb"],
        "template_ccd": best["ccd"],
        "mcs_atoms": best["mcs_atoms"],
        "mcs_coverage": best["mcs_atoms"] / qha if qha else None,
    }
    mcs_smarts = best["mcs_smarts"]

    # Reference frame = model_0 (no transform needed for model_0)
    ref_ca, ref_res = boltz_ca, boltz_res

    candidate_pdbs: dict[str, Path] = {
        f"boltz_model_{i}": boltz_dir.joinpath(f"{qid}_model_{i}.pdb") for i in range(5)
    }
    candidate_pdbs["gnina_refined"] = GNINA_DIR.joinpath(qid, f"{qid}_refined.pdb")
    candidate_pdbs["template_transferred"] = TEMPLATE_DIR.joinpath(
        qid, f"{qid}_template.pdb"
    )

    for source, pdb_path in candidate_pdbs.items():
        row = {**base, "source": source}
        if not pdb_path.exists():
            row["error"] = "pdb_missing"
            rows.append(row)
            continue
        try:
            if source == "boltz_model_0":
                R_c, t_c = np.eye(3), np.zeros(3)
            elif source.startswith("boltz_model_"):
                R_c, t_c = _superpose_pdb_to_ref(pdb_path, ref_ca, ref_res)
            else:
                # gnina_refined and template_transferred are already in model_0 frame
                R_c, t_c = np.eye(3), np.zeros(3)
            cand_mol = _transform_ligand_in_pdb(pdb_path, qsmi, R_c, t_c)
            row["template_rmsd"] = _mcs_rmsd(cand_mol, template_mol, mcs_smarts)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-mcs",
        type=int,
        default=8,
        help="Minimum MCS heavy atoms to bother scoring a query (default 8).",
    )
    parser.add_argument("--templates-csv", type=Path, default=TEMPLATE_DB_CSV)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    templates_df = (
        pd.read_csv(args.templates_csv).drop_duplicates("smiles").reset_index(drop=True)
    )
    print(
        f"Queries: {len(df)} | Unique-smiles templates: {len(templates_df)} | "
        f"min_mcs={args.min_mcs}"
    )

    all_rows: list[dict[str, Any]] = []
    for n, (_, row) in enumerate(df.iterrows(), 1):
        all_rows.extend(
            _process_one(row.structure, row.smiles, templates_df, args.min_mcs)
        )
        if n % 20 == 0 or n == len(df):
            print(f"  {n}/{len(df)}")

    out_df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out_df)} rows)")

    # Summary
    if "skipped" in out_df.columns:
        skipped = out_df[out_df["skipped"].notna()]["compound"].nunique()
        print(f"Skipped (mcs<{args.min_mcs}): {skipped}")
    if "template_rmsd" in out_df.columns:
        scored = out_df[out_df["template_rmsd"].notna()].copy()
        if not scored.empty:
            print("\nTemplate-RMSD by source (Å):")
            agg = scored.groupby("source")["template_rmsd"].agg(
                ["mean", "median", "count"]
            )
            print(agg.round(3).to_string())

            print("\nWhich source has the lowest template-RMSD per query?")
            best_per_query = (
                scored.sort_values("template_rmsd")
                .groupby("compound")
                .first()
                .reset_index()
            )
            print(best_per_query["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
