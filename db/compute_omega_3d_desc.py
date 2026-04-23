"""Compute 3D descriptors on OpenEye Omega conformers (lowest-energy per mol).

Uses the same descriptor definitions as ``compound_boltz2_desc3d`` and
``compound_boltz2_desc3d_vector``, but reads from ``structures/omega/<id>.sdf``
(multi-conformer, first conformer = lowest-energy per Omega sort).

Boltz-2 poses are docked bound-state conformers; Omega ensembles are
statistical solution-state conformers. Same descriptor formula on
different conformer source = genuinely new feature axis.

Output tables:
  compound_omega_desc3d           (scalar 11)
  compound_omega_desc3d_vector    (RDKit vector suite: 80+273+224+210+114+12+60 = 973d)

Skipped: scikit-fingerprints 3D (e3fp/electroshape) — needs separate venv.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch
from rdkit import Chem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.rdMolDescriptors import CalcPBF

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OMEGA_DIR = REPO_ROOT.joinpath("structures", "omega")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS compound_omega_desc3d (
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
    computed_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS compound_omega_desc3d_vector (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    autocorr3d JSONB,
    getaway JSONB,
    morse JSONB,
    rdf JSONB,
    whim JSONB,
    usr JSONB,
    usrcat JSONB,
    computed_at TIMESTAMPTZ DEFAULT now()
);
"""

SCALAR_COLS = (
    "compound_id",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pmi1",
    "pmi2",
    "pmi3",
    "radius_of_gyration",
    "spherocity_index",
    "pbf",
)
VECTOR_COLS = (
    "compound_id",
    "autocorr3d",
    "getaway",
    "morse",
    "rdf",
    "whim",
    "usr",
    "usrcat",
)


def load_first_conformer(sdf_path: Path):
    """Load the first (lowest-energy) conformer from a multi-conformer SDF."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for mol in suppl:
        if mol is None:
            continue
        return mol
    return None


def compute_scalar(mol) -> tuple | None:
    try:
        return (
            Descriptors3D.Asphericity(mol),
            Descriptors3D.Eccentricity(mol),
            Descriptors3D.InertialShapeFactor(mol),
            Descriptors3D.NPR1(mol),
            Descriptors3D.NPR2(mol),
            Descriptors3D.PMI1(mol),
            Descriptors3D.PMI2(mol),
            Descriptors3D.PMI3(mol),
            Descriptors3D.RadiusOfGyration(mol),
            Descriptors3D.SpherocityIndex(mol),
            CalcPBF(mol),
        )
    except Exception:
        return None


def compute_vector(mol) -> dict | None:
    try:
        return {
            "autocorr3d": json.dumps(list(rdMD.CalcAUTOCORR3D(mol))),
            "getaway": json.dumps(list(rdMD.CalcGETAWAY(mol))),
            "morse": json.dumps(list(rdMD.CalcMORSE(mol))),
            "rdf": json.dumps(list(rdMD.CalcRDF(mol))),
            "whim": json.dumps(list(rdMD.CalcWHIM(mol))),
            "usr": json.dumps(list(rdMD.GetUSR(mol))),
            "usrcat": json.dumps(list(rdMD.GetUSRCAT(mol))),
        }
    except Exception:
        return None


def main() -> None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        for stmt in SCHEMA_SQL.split(";"):
            if stmt.strip():
                cur.execute(stmt)
        conn.commit()

    # List compound ids that have omega SDFs and need compute
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT compound_id FROM compound_omega3d WHERE status = 'ok'
            AND compound_id NOT IN (
                SELECT compound_id FROM compound_omega_desc3d
            )
            ORDER BY compound_id
            """
        )
        todo = [r[0] for r in cur.fetchall()]
    print(f"To compute: {len(todo)} compounds")

    t0 = time.time()
    scalar_rows = []
    vector_rows = []
    failed = 0

    for i, cid in enumerate(todo, start=1):
        sdf = OMEGA_DIR.joinpath(f"{cid:05d}.sdf")
        mol = load_first_conformer(sdf)
        if mol is None:
            failed += 1
            continue

        s = compute_scalar(mol)
        v = compute_vector(mol)

        if s is not None:
            scalar_rows.append((cid, *s))
        if v is not None:
            vector_rows.append(
                (
                    cid,
                    v["autocorr3d"],
                    v["getaway"],
                    v["morse"],
                    v["rdf"],
                    v["whim"],
                    v["usr"],
                    v["usrcat"],
                )
            )

        if i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(todo) - i) / rate
            print(f"  {i}/{len(todo)} rate={rate:.1f}/s eta={eta:.0f}s failed={failed}")
            # Flush
            _flush(scalar_rows, vector_rows)
            scalar_rows = []
            vector_rows = []

    _flush(scalar_rows, vector_rows)
    print(f"\nFinished in {time.time() - t0:.0f}s  failed={failed}")


def _flush(scalar_rows, vector_rows) -> None:
    if not scalar_rows and not vector_rows:
        return
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        if scalar_rows:
            execute_batch(
                cur,
                f"INSERT INTO compound_omega_desc3d ({','.join(SCALAR_COLS)}) "
                f"VALUES ({','.join(['%s'] * len(SCALAR_COLS))}) "
                f"ON CONFLICT (compound_id) DO UPDATE SET "
                + ", ".join(
                    f"{c} = EXCLUDED.{c}" for c in SCALAR_COLS if c != "compound_id"
                ),
                scalar_rows,
            )
        if vector_rows:
            execute_batch(
                cur,
                f"INSERT INTO compound_omega_desc3d_vector ({','.join(VECTOR_COLS)}) "
                f"VALUES ({','.join(['%s'] * len(VECTOR_COLS))}) "
                f"ON CONFLICT (compound_id) DO UPDATE SET "
                + ", ".join(
                    f"{c} = EXCLUDED.{c}" for c in VECTOR_COLS if c != "compound_id"
                ),
                vector_rows,
            )
        conn.commit()


if __name__ == "__main__":
    main()
