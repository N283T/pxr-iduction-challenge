"""Phase 2 post-processing utilities for the Boltz-2 PXR pipeline.

This module is side-effect free: it reads predicted complexes and the
ligand mol cache that Boltz writes during preprocessing, builds a fully
bonded RDKit Mol with the predicted pose coordinates, and returns
metadata dicts. The actual filesystem writes and DB upserts live in
``track1_activity/boltz2/scripts/boltz2_postprocess.py`` so this module can be
imported and unit-tested without side effects.

Why we need the cached pickle
-----------------------------
The mmCIF that ``boltz predict`` produces only contains atom positions
for the ligand, no bond block. Re-perceiving bonds from coordinates plus
``AllChem.AssignBondOrdersFromTemplate`` works for the skeleton but loses
H counting and aromaticity flags. The Boltz preprocessing step pickles
the original RDKit Mol it builds from the input SMILES (with correct
bonds, aromaticity and H counts) under
``boltz_results_<input>/processed/mols/<id>.pkl``. We re-use that mol
verbatim and only swap its 3D coordinates with the ones from the
predicted cif. The atom order between the pickled mol and the cif
ligand chain is preserved by Boltz, so the swap is one-to-one.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from .constants import (
    LIGAND_CHAIN_ID,
    PROTEIN_CHAIN_ID,
    PXR_CORE_POCKET_RESIDUES,
)


# Suppress noisy info / warning logs but keep RDKit errors visible.
RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")


# ---------------------------------------------------------------------------
# Pose mol construction
# ---------------------------------------------------------------------------


def load_template_mol(pkl_path: Path) -> Chem.Mol:
    """Load the cached RDKit Mol that Boltz produced from the input SMILES.

    The pickle stores ``{ccd_code: rdkit.Chem.Mol}``. PXR pipeline always
    has a single ligand per compound, so we return that single value.
    """
    with pkl_path.open("rb") as f:
        extras = pickle.load(f)
    if not isinstance(extras, dict) or len(extras) == 0:
        raise ValueError(
            f"unexpected pkl content at {pkl_path}: {type(extras).__name__}"
        )
    if len(extras) > 1:
        raise ValueError(
            f"expected exactly one ligand in {pkl_path}, got {len(extras)}: "
            f"{list(extras)}"
        )
    return next(iter(extras.values()))


def read_ligand_atoms_from_cif(
    cif_path: Path, ligand_chain: str = LIGAND_CHAIN_ID
) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Return ``(coords, element_symbols)`` for the ligand chain in a cif."""
    structure = gemmi.read_structure(str(cif_path))
    chain = structure[0].find_chain(ligand_chain)
    if chain is None:
        raise ValueError(f"ligand chain {ligand_chain!r} missing in {cif_path}")
    coords: list[tuple[float, float, float]] = []
    symbols: list[str] = []
    for residue in chain:
        for atom in residue:
            coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
            symbols.append(atom.element.name.capitalize())
    return coords, symbols


def build_pose_mol(pkl_path: Path, cif_path: Path) -> Chem.Mol:
    """Combine the cached template mol with the predicted pose coordinates.

    The returned mol has all bond orders, aromaticity flags and H counts
    from the original SMILES, but its single conformer carries the 3D
    coordinates that Boltz predicted. Stereo (R/S, E/Z) is re-perceived
    from the new 3D coordinates so any chirality the model assigned is
    reflected.
    """
    template = load_template_mol(pkl_path)
    coords, cif_symbols = read_ligand_atoms_from_cif(cif_path)
    template_symbols = [a.GetSymbol() for a in template.GetAtoms()]
    if template_symbols != cif_symbols:
        raise ValueError(
            "atom symbol order mismatch between cached pkl and predicted cif "
            f"({pkl_path.name} vs {cif_path.name})"
        )

    pose = Chem.Mol(template)
    pose.RemoveAllConformers()
    conf = Chem.Conformer(pose.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    pose.AddConformer(conf, assignId=True)

    # Re-perceive stereo from the predicted 3D coordinates. R/S
    # determination needs the implicit H positions, so we briefly
    # add explicit Hs (with template-based 3D coordinates), let RDKit
    # set the chiral tags, then strip the Hs again before returning.
    pose_with_h = Chem.AddHs(pose, addCoords=True)
    Chem.AssignStereochemistryFrom3D(pose_with_h)
    return Chem.RemoveHs(pose_with_h)


def write_pose_files(mol: Chem.Mol, pkl_out: Path, sdf_out: Path) -> None:
    """Save the pose mol as both an RDKit pickle and an SDF for viewers."""
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    sdf_out.parent.mkdir(parents=True, exist_ok=True)

    with pkl_out.open("wb") as f:
        pickle.dump(mol, f)

    writer = Chem.SDWriter(str(sdf_out))
    writer.SetKekulize(False)  # write aromatic bonds as bond type 4
    writer.write(mol)
    writer.close()


# ---------------------------------------------------------------------------
# Geometry sanity check
# ---------------------------------------------------------------------------


def _pocket_centroid(
    model: gemmi.Model, residue_numbers: tuple[int, ...]
) -> np.ndarray:
    chain = model.find_chain(PROTEIN_CHAIN_ID)
    if chain is None:
        raise ValueError(f"protein chain {PROTEIN_CHAIN_ID!r} missing")
    target = set(residue_numbers)
    coords: list[tuple[float, float, float]] = []
    for residue in chain:
        if residue.seqid.num in target:
            for atom in residue:
                if atom.name == "CA":
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    if not coords:
        raise ValueError("none of the requested pocket residues were found")
    return np.asarray(coords).mean(axis=0)


def compute_geometry_metrics(
    cif_path: Path,
    ligand_chain: str = LIGAND_CHAIN_ID,
    pocket_residues: tuple[int, ...] = PXR_CORE_POCKET_RESIDUES,
) -> dict[str, float | int]:
    """Return ligand atom count and ligand centroid to pocket centroid distance."""
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    ligand = model.find_chain(ligand_chain)
    if ligand is None:
        return {"ligand_atom_count": 0, "ligand_to_pocket_distance_a": float("nan")}
    ligand_coords = [
        (atom.pos.x, atom.pos.y, atom.pos.z) for residue in ligand for atom in residue
    ]
    if not ligand_coords:
        return {"ligand_atom_count": 0, "ligand_to_pocket_distance_a": float("nan")}
    ligand_arr = np.asarray(ligand_coords)
    pocket = _pocket_centroid(model, pocket_residues)
    distance = float(np.linalg.norm(ligand_arr.mean(axis=0) - pocket))
    return {
        "ligand_atom_count": int(len(ligand_coords)),
        "ligand_to_pocket_distance_a": distance,
    }


# ---------------------------------------------------------------------------
# JSON metric loaders
# ---------------------------------------------------------------------------


def load_scalar_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and keep only top-level scalar entries."""
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, (int, float, bool, str))}


# ---------------------------------------------------------------------------
# Per-compound aggregator
# ---------------------------------------------------------------------------


# Boltz output filename templates relative to predictions_dir/<cid>/
PREDICTION_FILE_TEMPLATES = {
    "pose_cif_path": "{cid}_model_0.cif",
    "confidence_json_path": "confidence_{cid}_model_0.json",
    "affinity_json_path": "affinity_{cid}.json",
    "plddt_npz_path": "plddt_{cid}_model_0.npz",
    "pae_npz_path": "pae_{cid}_model_0.npz",
    "pde_npz_path": "pde_{cid}_model_0.npz",
    "embeddings_npz_path": "embeddings_{cid}.npz",
}

CONFIDENCE_KEYS = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)

# Boltz writes affinity_pred_value, affinity_pred_value1, affinity_pred_value2
# (no underscore before the index). We map them to SQL-friendly snake names.
AFFINITY_KEY_MAP = {
    "affinity_pred_value": "affinity_pred_value",
    "affinity_probability_binary": "affinity_probability_binary",
    "affinity_pred_value1": "affinity_pred_value_1",
    "affinity_probability_binary1": "affinity_probability_binary_1",
    "affinity_pred_value2": "affinity_pred_value_2",
    "affinity_probability_binary2": "affinity_probability_binary_2",
}


def collect_compound_metadata(
    predictions_dir: Path,
    mols_dir: Path,
    pose_out_dir: Path,
    compound_id: int,
) -> dict[str, Any]:
    """Build the full metadata dict for one compound's Boltz-2 prediction.

    The dict shape matches the columns of the ``compound_boltz2`` table.
    Compounds whose preprocessing failed (no prediction directory) get
    ``preprocessing_failed=True`` and a ``failure_reason``; their file
    path columns stay NULL.
    """
    cid_str = f"{compound_id:05d}"
    compound_dir = predictions_dir.joinpath(cid_str)

    record: dict[str, Any] = {
        "compound_id": compound_id,
        "preprocessing_failed": False,
        "failure_reason": None,
        "ligand_oversize": False,
    }

    if not compound_dir.is_dir():
        record["preprocessing_failed"] = True
        record["failure_reason"] = "no prediction directory"
        return record

    # Resolve standard prediction file paths.
    for key, template in PREDICTION_FILE_TEMPLATES.items():
        path = compound_dir.joinpath(template.format(cid=cid_str))
        record[key] = str(path) if path.exists() else None

    pkl_in = mols_dir.joinpath(f"{cid_str}.pkl")
    cif_in = compound_dir.joinpath(f"{cid_str}_model_0.cif")
    if not pkl_in.exists() or not cif_in.exists():
        record["preprocessing_failed"] = True
        record["failure_reason"] = "missing pkl or cif"
        return record

    # Build pose mol and write pkl + sdf.
    try:
        pose = build_pose_mol(pkl_in, cif_in)
        pose_pkl_out = pose_out_dir.joinpath(f"{cid_str}.pkl")
        pose_sdf_out = pose_out_dir.joinpath(f"{cid_str}.sdf")
        write_pose_files(pose, pose_pkl_out, pose_sdf_out)
        record["ligand_pkl_path"] = str(pose_pkl_out)
        record["ligand_sdf_path"] = str(pose_sdf_out)
    except Exception as exc:  # noqa: BLE001 — record and continue
        record["preprocessing_failed"] = True
        record["failure_reason"] = f"pose build failed: {type(exc).__name__}: {exc}"
        return record

    # Geometry sanity.
    geom = compute_geometry_metrics(cif_in)
    record.update(geom)
    if record.get("ligand_atom_count", 0) > 56:
        record["ligand_oversize"] = True

    # Confidence metrics.
    confidence = load_scalar_json(
        compound_dir.joinpath(f"confidence_{cid_str}_model_0.json")
    )
    for key in CONFIDENCE_KEYS:
        record[key] = confidence.get(key)

    # Affinity metrics.
    affinity = load_scalar_json(compound_dir.joinpath(f"affinity_{cid_str}.json"))
    for boltz_key, sql_key in AFFINITY_KEY_MAP.items():
        record[sql_key] = affinity.get(boltz_key)

    return record
