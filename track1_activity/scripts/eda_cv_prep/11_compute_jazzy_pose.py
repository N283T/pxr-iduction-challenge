"""Compute Jazzy free-energy descriptors on the Boltz-2 pose conformer.

The existing `compound_jazzy` table holds 6 Jazzy values computed on
Jazzy's self-generated 3D conformer (ETKDG + MMFF94 minimisation).
This script re-computes the same 6 values on the Boltz-2 predicted
binding pose so the signal reflects "free energy vector in the
predicted binding geometry" rather than "free energy vector in a
solvated-and-relaxed standalone geometry".

Output: compound_boltz2_jazzy (one row per compound with successful
pose, 6 double columns + computed_at).

Bypasses `molecular_vector_from_smiles` (which insists on generating
its own conformer) and calls the underlying Jazzy functions directly
with the pose mol.
"""

from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch
from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402

# Jazzy low-level API
from jazzy.api import (
    calculate_delta_apolar,
    calculate_delta_interaction,
    calculate_delta_polar,
    calculate_polar_strength_map,
    config as jcfg,  # Config instance with fitted coefficients
    get_charges_from_kallisto_molecule,
    get_covalent_atom_idxs,
    kallisto_molecule_from_rdkit_molecule,
    sum_atomic_map,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS compound_boltz2_jazzy (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    sdc   DOUBLE PRECISION,
    sdx   DOUBLE PRECISION,
    sa    DOUBLE PRECISION,
    dga   DOUBLE PRECISION,
    dgp   DOUBLE PRECISION,
    dgtot DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);
"""

FEATURE_COLS = ("sdc", "sdx", "sa", "dga", "dgp", "dgtot")


def jazzy_vector_from_mol(mol: Chem.Mol) -> dict:
    """Mirror `molecular_vector_from_smiles` but skip conformer generation.

    The mol passed in must already have a 3D conformer (Boltz-2 pose).
    The pose pkls are stored post-``RemoveHs``, so we add Hs back with
    ``addCoords=True`` (places Hs consistent with the heavy-atom pose
    geometry). Without this the sdc/sdx donor-strength sums are
    always 0 (H atoms not visible to Jazzy's atom iteration).
    """
    mol = Chem.AddHs(mol, addCoords=True)
    kalli = kallisto_molecule_from_rdkit_molecule(mol)
    atoms_and_nbrs = get_covalent_atom_idxs(mol)
    charges = get_charges_from_kallisto_molecule(kalli, 0)
    atomic_map = calculate_polar_strength_map(mol, kalli, atoms_and_nbrs, charges)
    vec = sum_atomic_map(atomic_map)

    dga = calculate_delta_apolar(
        mol, atomic_map, jcfg.g0, jcfg.gs, jcfg.gr, jcfg.gpi1, jcfg.gpi2
    )
    dgp = calculate_delta_polar(
        atomic_map, atoms_and_nbrs, jcfg.gd, jcfg.ga, jcfg.expd, jcfg.expa
    )
    dgi = calculate_delta_interaction(
        mol, atomic_map, atoms_and_nbrs, jcfg.gi, jcfg.expa, jcfg.f
    )
    vec["dga"] = dga
    vec["dgp"] = dgp
    vec["dgtot"] = dga + dgp + dgi
    return vec


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    conn.commit()

    cur.execute(
        """
        SELECT compound_id, ligand_pkl_path
        FROM compound_boltz2
        WHERE ligand_pkl_path IS NOT NULL
        ORDER BY compound_id
        """
    )
    rows = cur.fetchall()
    print(f"Candidates with PKL: {len(rows)}")

    cur.execute("SELECT compound_id FROM compound_boltz2_jazzy")
    already = {r[0] for r in cur.fetchall()}
    todo = [(cid, p) for cid, p in rows if cid not in already]
    print(f"Already done: {len(already)}, to compute: {len(todo)}")
    if not todo:
        return

    batch: list[tuple] = []
    failed: list[tuple[int, str]] = []
    t0 = time.time()
    for i, (cid, pkl_path) in enumerate(todo):
        try:
            with open(pkl_path, "rb") as f:
                mol = pickle.load(f)
            if not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
                failed.append((cid, "no conformer"))
                continue
            vec = jazzy_vector_from_mol(mol)
            batch.append((cid, *(float(vec[c]) for c in FEATURE_COLS)))
        except Exception as exc:  # noqa: BLE001
            failed.append((cid, f"{type(exc).__name__}: {exc}"))

        if len(batch) >= 250 or i + 1 == len(todo):
            execute_batch(
                cur,
                """
                INSERT INTO compound_boltz2_jazzy
                  (compound_id, sdc, sdx, sa, dga, dgp, dgtot)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (compound_id) DO UPDATE SET
                    sdc   = EXCLUDED.sdc,
                    sdx   = EXCLUDED.sdx,
                    sa    = EXCLUDED.sa,
                    dga   = EXCLUDED.dga,
                    dgp   = EXCLUDED.dgp,
                    dgtot = EXCLUDED.dgtot,
                    computed_at = NOW()
                """,
                batch,
            )
            conn.commit()
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(todo) - (i + 1)) / max(rate, 1e-6)
            print(
                f"  [{i + 1}/{len(todo)}] flushed {len(batch)} rows  "
                f"rate={rate:.1f}/s  ETA={eta:.0f}s"
            )
            batch = []

    print(f"\nDone. Failed: {len(failed)}")
    for cid, reason in failed[:10]:
        print(f"  {cid}: {reason}")

    cur.execute("SELECT COUNT(*) FROM compound_boltz2_jazzy")
    n = cur.fetchone()[0]
    print(f"Rows in compound_boltz2_jazzy: {n}")
    conn.close()


if __name__ == "__main__":
    main()
