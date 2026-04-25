"""Boltz-2 input YAML builder for the Track 2 (Structure Prediction) pipeline.

Layout per YAML follows the official tutorial:
- monomeric LBD-only PXR sequence in chain A (no MSA, no templates),
- optional ligand SMILES in chain B,
- optional affinity property requested for the ligand (free side-channel data).

We deliberately omit the ``msa:`` field so that ``boltz predict --use_msa_server``
can fetch a fresh ColabFold MSA for the LBD sequence and cache it. Pocket
constraints are also intentionally absent — fragments make up ~35% of the
Track 2 set and we don't want to bias their pose toward the Track 1 binding-site
residue list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import LIGAND_CHAIN_ID, PROTEIN_CHAIN_ID, PXR_LBD_SEQUENCE


def build_yaml(
    smiles: str | None,
    *,
    sequence: str = PXR_LBD_SEQUENCE,
    request_affinity: bool = True,
) -> dict[str, Any]:
    """Build a Boltz-2 YAML dict for an apo or holo prediction.

    Pass ``smiles=None`` to produce an apo (ligand-free) input — useful as
    a one-off run to populate the MSA cache and produce a reference apo
    LBD model for analysis.
    """
    sequences: list[dict[str, Any]] = [
        {"protein": {"id": PROTEIN_CHAIN_ID, "sequence": sequence}},
    ]
    if smiles is not None:
        sequences.append({"ligand": {"id": LIGAND_CHAIN_ID, "smiles": smiles}})

    yaml_dict: dict[str, Any] = {"version": 1, "sequences": sequences}

    if smiles is not None and request_affinity:
        yaml_dict["properties"] = [{"affinity": {"binder": LIGAND_CHAIN_ID}}]

    return yaml_dict


def write_yaml(path: Path, smiles: str | None, **kwargs: Any) -> None:
    """Write a Boltz-2 YAML file to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_dict = build_yaml(smiles, **kwargs)
    path.write_text(yaml.safe_dump(yaml_dict, sort_keys=False))
