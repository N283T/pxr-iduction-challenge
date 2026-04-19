"""Compute Mordred 3D descriptors from Boltz-2 poses.

Adds the 213 Mordred descriptors that require a 3D conformer, on top
of the 2D-only compound_mordred already in the DB (that table was
computed with ignore_3D=True per db/compute_mordred.py).

Mordred wraps RDKit under the hood, so large portions of the 213
duplicate the scalar3d + vector3d tables computed earlier today. The
unique contribution is Mordred's specific aggregation / normalisation
choices (e.g. GravitationalIndex, MomentOfInertia variants beyond
PMI1-3). Keeping them all and letting feature-importance / downstream
modelling decide is cheaper than curating by hand.

Output: compound_boltz2_mordred3d (compound_id PK, descriptors JSONB).

Uses multiprocessing (nprocs=8) to parallelise across compounds.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
from mordred import Calculator, descriptors
from psycopg2.extras import execute_batch
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS compound_boltz2_mordred3d (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    descriptors JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def build_3d_calc() -> Calculator:
    """Return a Calculator containing only Mordred's 3D-requiring descriptors."""
    calc_all = Calculator(descriptors.all, ignore_3D=False)
    calc_2d = Calculator(descriptors.all, ignore_3D=True)
    names_2d = {str(d) for d in calc_2d.descriptors}
    d3_only = [d for d in calc_all.descriptors if str(d) not in names_2d]
    return Calculator(d3_only, ignore_3D=False)


def load_pose(pkl_path: str) -> Chem.Mol | None:
    try:
        with open(pkl_path, "rb") as f:
            mol = pickle.load(f)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
        return None
    # Mordred 3D descriptors operate on heavy-atom geometry; pose pkls were
    # stored post-RemoveHs. Mordred does not require explicit Hs for its
    # 3D descriptors (distinct from Jazzy which needs Hs for H-bond terms).
    return mol


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

    cur.execute("SELECT compound_id FROM compound_boltz2_mordred3d")
    already = {r[0] for r in cur.fetchall()}
    todo = [(cid, p) for cid, p in rows if cid not in already]
    print(f"Already done: {len(already)}, to compute: {len(todo)}")
    if not todo:
        return

    print("Building Mordred 3D-only calculator...")
    calc = build_3d_calc()
    print(f"  3D descriptors: {len(calc)}")

    print("Loading pose mols...")
    t0 = time.time()
    ids, mols = [], []
    for cid, pkl in todo:
        mol = load_pose(pkl)
        if mol is None:
            print(f"  skip {cid}: invalid pose")
            continue
        ids.append(cid)
        mols.append(mol)
    print(f"  loaded {len(mols)} mols in {time.time() - t0:.1f}s")

    print("Running Mordred (parallel, nproc=8)...")
    t1 = time.time()
    results = list(calc.map(mols, nproc=8, quiet=False))
    print(f"  Mordred done in {time.time() - t1:.1f}s")

    print("Writing to DB...")
    batch = []
    names = [str(d) for d in calc.descriptors]
    for cid, row in zip(ids, results):
        # row is a mordred Result; iterate or index
        d = {}
        for name, val in zip(names, row):
            try:
                f = float(val)
                if not np.isfinite(f):
                    f = None
            except Exception:  # noqa: BLE001
                f = None
            d[name] = f
        batch.append((cid, json.dumps(d)))

    execute_batch(
        cur,
        """
        INSERT INTO compound_boltz2_mordred3d (compound_id, descriptors)
        VALUES (%s, %s)
        ON CONFLICT (compound_id) DO UPDATE SET
            descriptors = EXCLUDED.descriptors,
            computed_at = NOW()
        """,
        batch,
        page_size=200,
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM compound_boltz2_mordred3d")
    n = cur.fetchone()[0]
    print(f"\nRows in compound_boltz2_mordred3d: {n}")
    conn.close()


if __name__ == "__main__":
    main()
