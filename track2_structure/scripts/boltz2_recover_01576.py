"""Recovery script for compound 01576 — largest fragment of a 2-component salt.

Compound 01576's ``std_smiles`` is a 2-fragment complex (62 + 61 heavy
atoms, both avermectin/milbemycin-like macrolides). ChEMBL structure
pipeline ``get_parent_mol`` cannot pick a single parent because neither
fragment is small enough to be treated as a counter-ion, so the raw
``A.B`` SMILES is fed to Boltz-2 and the preprocessing step crashes
inside ``LargestFragmentChooser``.

This script picks the larger fragment with RDKit's
``rdMolStandardize.LargestFragmentChooser``, overwrites
``structures/boltz2/inputs/01576.yaml`` with the new SMILES, and
exits. The caller then re-runs ``boltz2_full_run.sh`` to let Boltz
process the one newly-missing input.

Rationale, caveats (both fragments are comparable size, so "largest"
is an arbitrary tiebreaker), and the post-recovery pose / affinity
values are tracked in GitHub issue #50.

Usage
-----
    pixi run python track2_structure/scripts/boltz2_recover_01576.py
    bash track2_structure/scripts/boltz2_full_run.sh  # re-run inference
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.joinpath("src")))

from boltz2.constants import INPUTS_DIR  # noqa: E402
from boltz2.input_builder import build_yaml, write_yaml  # noqa: E402


COMPOUND_ID = 1576


def main() -> None:
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT std_smiles FROM compounds WHERE id = %s", (COMPOUND_ID,)
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"compound {COMPOUND_ID} not found in compounds table")
    (std_smiles,) = row

    mol = Chem.MolFromSmiles(std_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse std_smiles for {COMPOUND_ID}")

    frags = Chem.GetMolFrags(mol, asMols=True)
    largest = rdMolStandardize.LargestFragmentChooser().choose(mol)
    recovery_smiles = Chem.MolToSmiles(largest)

    print(f"[recover-01576] original heavy atoms: {mol.GetNumAtoms()}")
    print(f"[recover-01576] fragments           : {len(frags)}")
    for i, frag in enumerate(frags):
        print(
            f"[recover-01576]   frag_{i}: {frag.GetNumAtoms():>3d} HAs, "
            f"MW {Descriptors.MolWt(frag):.1f}"
        )
    print(f"[recover-01576] largest HAs         : {largest.GetNumAtoms()}")
    print(f"[recover-01576] largest MW          : {Descriptors.MolWt(largest):.1f}")
    print(
        f"[recover-01576] oversize (>56 HAs)  : {largest.GetNumAtoms() > 56} "
        "(affinity head trained only up to 56 HAs)"
    )
    print(f"[recover-01576] recovery SMILES     : {recovery_smiles}")

    yaml_dict = build_yaml(recovery_smiles)
    yaml_path = INPUTS_DIR.joinpath(f"{COMPOUND_ID:05d}.yaml")
    write_yaml(yaml_dict, yaml_path)
    print(f"[recover-01576] overwrote           : {yaml_path}")
    print(
        "[recover-01576] next step            : "
        "bash track2_structure/scripts/boltz2_full_run.sh"
    )


if __name__ == "__main__":
    main()
