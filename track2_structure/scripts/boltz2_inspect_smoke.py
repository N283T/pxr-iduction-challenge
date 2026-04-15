"""Inspect Boltz-2 smoke test outputs.

Reads each predicted complex under ``structures/boltz2/outputs_smoke/`` and
reports per-compound:

- whether the mmCIF parses
- which chains are present
- the ligand atom count
- the distance between the ligand centroid and the PXR core pocket centroid
- the affinity prediction value (when present)
- selected confidence metrics

This is the manual gate after running ``boltz2_smoke_run.sh``: if the values
look reasonable across the 10 smoke compounds, the same parameters can be
scaled up to the full 4653 compound run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.joinpath("src")))

from boltz2.constants import (  # noqa: E402
    BOLTZ2_DIR,
    LIGAND_CHAIN_ID,
    OUTPUTS_SMOKE_DIR,
    PROTEIN_CHAIN_ID,
    PXR_CORE_POCKET_RESIDUES,
)


def find_predictions_root(out_dir: Path) -> Path:
    """Locate the ``boltz_results_<input>/predictions/`` subdirectory."""
    candidates = sorted(out_dir.glob("boltz_results_*/predictions"))
    if not candidates:
        raise FileNotFoundError(
            f"No boltz_results_*/predictions directory found under {out_dir}"
        )
    return candidates[0]


def chain_centroid(model: gemmi.Model, chain_id: str) -> np.ndarray | None:
    """Mean position of all atoms in a chain (None if chain missing/empty)."""
    chain = model.find_chain(chain_id)
    if chain is None:
        return None
    coords: list[tuple[float, float, float]] = []
    for residue in chain:
        for atom in residue:
            coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    if not coords:
        return None
    return np.asarray(coords).mean(axis=0)


def pocket_centroid(
    model: gemmi.Model,
    residue_numbers: tuple[int, ...],
    chain_id: str = PROTEIN_CHAIN_ID,
) -> np.ndarray:
    """Centroid of CA atoms over the given residue numbers."""
    chain = model.find_chain(chain_id)
    if chain is None:
        raise ValueError(f"Protein chain {chain_id!r} not found in model")
    target = set(residue_numbers)
    coords: list[tuple[float, float, float]] = []
    for residue in chain:
        if residue.seqid.num in target:
            for atom in residue:
                if atom.name == "CA":
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    if not coords:
        raise ValueError(
            f"None of the requested pocket residues found in chain {chain_id!r}"
        )
    return np.asarray(coords).mean(axis=0)


def count_chain_atoms(model: gemmi.Model, chain_id: str) -> int:
    chain = model.find_chain(chain_id)
    if chain is None:
        return 0
    return sum(1 for residue in chain for _ in residue)


def inspect_one(compound_dir: Path) -> dict[str, Any]:
    compound_id = compound_dir.name
    cif_files = sorted(compound_dir.glob(f"{compound_id}_model_*.cif"))
    confidence_files = sorted(
        compound_dir.glob(f"confidence_{compound_id}_model_*.json")
    )
    affinity_files = sorted(compound_dir.glob(f"affinity_{compound_id}.json"))

    record: dict[str, Any] = {
        "compound_id": compound_id,
        "cif_count": len(cif_files),
        "confidence_count": len(confidence_files),
        "affinity_count": len(affinity_files),
    }

    if not cif_files:
        record["status"] = "no_cif"
        return record

    try:
        structure = gemmi.read_structure(str(cif_files[0]))
        model = structure[0]
        record["chain_ids"] = ",".join(chain.name for chain in model)
        record["protein_atoms"] = count_chain_atoms(model, PROTEIN_CHAIN_ID)
        record["ligand_atoms"] = count_chain_atoms(model, LIGAND_CHAIN_ID)

        ligand_com = chain_centroid(model, LIGAND_CHAIN_ID)
        if ligand_com is None:
            record["status"] = "no_ligand_chain"
            return record
        pocket_com = pocket_centroid(model, PXR_CORE_POCKET_RESIDUES)
        record["ligand_to_pocket_distance_A"] = float(
            np.linalg.norm(ligand_com - pocket_com)
        )
    except Exception as exc:  # noqa: BLE001 — best-effort inspector
        record["status"] = "cif_parse_error"
        record["error"] = str(exc)
        return record

    if confidence_files:
        with confidence_files[0].open() as f:
            conf = json.load(f)
        for key, value in conf.items():
            if isinstance(value, (int, float, str, bool)):
                record[f"conf.{key}"] = value

    if affinity_files:
        with affinity_files[0].open() as f:
            aff = json.load(f)
        for key, value in aff.items():
            if isinstance(value, (int, float, str, bool)):
                record[f"affinity.{key}"] = value

    record["status"] = "ok"
    return record


def main() -> None:
    predictions_root = find_predictions_root(OUTPUTS_SMOKE_DIR)
    print(f"[inspect_smoke] predictions root: {predictions_root}")

    compound_dirs = sorted(p for p in predictions_root.iterdir() if p.is_dir())
    print(f"[inspect_smoke] compound dirs found: {len(compound_dirs)}")

    records = [inspect_one(d) for d in compound_dirs]

    df = pd.json_normalize(records)
    out_csv = BOLTZ2_DIR.joinpath("smoke_inspect.csv")
    df.to_csv(out_csv, index=False)
    print(f"[inspect_smoke] wrote {out_csv}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
