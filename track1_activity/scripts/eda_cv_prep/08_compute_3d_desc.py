"""Compute basic 3D shape descriptors from Boltz-2 ligand SDFs.

Reads 4652 poses from `structures/boltz2/ligands/<id>.sdf` (one per
`compound_boltz2.ligand_sdf_path`), computes 11 RDKit shape/geometry
descriptors, and stores them in `compound_boltz2_desc3d`.

Descriptors:
  asphericity, eccentricity, inertial_shape_factor,
  npr1, npr2, pmi1, pmi2, pmi3,
  radius_of_gyration, spherocity_index, pbf
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch
from rdkit import Chem
from rdkit.Chem import Descriptors3D
from rdkit.Chem.rdMolDescriptors import CalcPBF

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS compound_boltz2_desc3d (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    asphericity DOUBLE PRECISION,
    eccentricity DOUBLE PRECISION,
    inertial_shape_factor DOUBLE PRECISION,
    npr1 DOUBLE PRECISION,
    npr2 DOUBLE PRECISION,
    pmi1 DOUBLE PRECISION,
    pmi2 DOUBLE PRECISION,
    pmi3 DOUBLE PRECISION,
    radius_of_gyration DOUBLE PRECISION,
    spherocity_index DOUBLE PRECISION,
    pbf DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def compute_3d(mol: Chem.Mol) -> dict:
    """Compute RDKit 3D descriptors on a mol with explicit conformer."""
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformer")

    out = {
        "asphericity": Descriptors3D.Asphericity(mol),
        "eccentricity": Descriptors3D.Eccentricity(mol),
        "inertial_shape_factor": Descriptors3D.InertialShapeFactor(mol),
        "npr1": Descriptors3D.NPR1(mol),
        "npr2": Descriptors3D.NPR2(mol),
        "pmi1": Descriptors3D.PMI1(mol),
        "pmi2": Descriptors3D.PMI2(mol),
        "pmi3": Descriptors3D.PMI3(mol),
        "radius_of_gyration": Descriptors3D.RadiusOfGyration(mol),
        "spherocity_index": Descriptors3D.SpherocityIndex(mol),
    }
    try:
        out["pbf"] = CalcPBF(mol)
    except Exception:
        out["pbf"] = None
    return out


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

    # Re-compute everyone (idempotent via ON CONFLICT); prior SDF pass left
    # kekulize-failed rows missing and partially-populated rows. PKL is
    # the lossless pose source per CLAUDE.md.
    to_do = rows
    print(f"To compute (overwriting any prior rows): {len(to_do)}")

    t0 = time.time()
    batch: list[tuple] = []
    failed: list[int] = []
    for i, (cid, pkl_path) in enumerate(to_do):
        try:
            with open(pkl_path, "rb") as f:
                mol = pickle.load(f)
            if not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
                failed.append(cid)
                continue
            d = compute_3d(mol)
            batch.append(
                (
                    cid,
                    d["asphericity"],
                    d["eccentricity"],
                    d["inertial_shape_factor"],
                    d["npr1"],
                    d["npr2"],
                    d["pmi1"],
                    d["pmi2"],
                    d["pmi3"],
                    d["radius_of_gyration"],
                    d["spherocity_index"],
                    d["pbf"],
                )
            )
        except Exception as exc:
            print(f"  {cid}: {exc}")
            failed.append(cid)

        if len(batch) >= 500 or i + 1 == len(to_do):
            execute_batch(
                cur,
                """
                INSERT INTO compound_boltz2_desc3d
                  (compound_id, asphericity, eccentricity, inertial_shape_factor,
                   npr1, npr2, pmi1, pmi2, pmi3, radius_of_gyration,
                   spherocity_index, pbf)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (compound_id) DO UPDATE SET
                    asphericity = EXCLUDED.asphericity,
                    eccentricity = EXCLUDED.eccentricity,
                    inertial_shape_factor = EXCLUDED.inertial_shape_factor,
                    npr1 = EXCLUDED.npr1,
                    npr2 = EXCLUDED.npr2,
                    pmi1 = EXCLUDED.pmi1,
                    pmi2 = EXCLUDED.pmi2,
                    pmi3 = EXCLUDED.pmi3,
                    radius_of_gyration = EXCLUDED.radius_of_gyration,
                    spherocity_index = EXCLUDED.spherocity_index,
                    pbf = EXCLUDED.pbf,
                    computed_at = NOW()
                """,
                batch,
            )
            conn.commit()
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(to_do) - (i + 1)) / max(rate, 1e-6)
            print(
                f"  [{i + 1}/{len(to_do)}] flushed {len(batch)} rows  "
                f"rate={rate:.1f}/s  ETA={eta:.0f}s"
            )
            batch = []

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        print(
            f"  failed compound_ids: {failed[:20]}{'...' if len(failed) > 20 else ''}"
        )

    cur.execute("SELECT COUNT(*) FROM compound_boltz2_desc3d")
    n_final = cur.fetchone()[0]
    print(f"Rows in compound_boltz2_desc3d: {n_final}")
    conn.close()


if __name__ == "__main__":
    main()
