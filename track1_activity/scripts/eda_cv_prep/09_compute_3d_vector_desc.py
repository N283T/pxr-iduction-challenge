"""Compute the full RDKit 3D vector-descriptor suite (973 dim total)
from Boltz-2 pose pkls.

Methods (all in rdkit.Chem.rdMolDescriptors):
  AUTOCORR3D (80), GETAWAY (273), MORSE (224), RDF (210), WHIM (114),
  USR (12), USRCAT (60)

Storage: one JSONB column per method in compound_boltz2_desc3d_vector.
If a method fails for a particular compound the column is stored as
NULL for that row. Script is idempotent via ON CONFLICT.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS compound_boltz2_desc3d_vector (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    autocorr3d JSONB,
    getaway JSONB,
    morse JSONB,
    rdf JSONB,
    whim JSONB,
    usr JSONB,
    usrcat JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def _to_list(x) -> list:
    arr = np.asarray(x, dtype=np.float64)
    # Replace non-finite with None inside the JSON payload so postgres
    # doesn't choke; downstream readers can coalesce to 0/NaN.
    arr = np.where(np.isfinite(arr), arr, None).tolist()
    return arr


def compute(mol: Chem.Mol) -> dict:
    """Compute all 7 vector 3D descriptors. Per-method failure => None."""
    out: dict[str, list | None] = {
        "autocorr3d": None,
        "getaway": None,
        "morse": None,
        "rdf": None,
        "whim": None,
        "usr": None,
        "usrcat": None,
    }
    for key, fn in [
        ("autocorr3d", rdMD.CalcAUTOCORR3D),
        ("getaway", rdMD.CalcGETAWAY),
        ("morse", rdMD.CalcMORSE),
        ("rdf", rdMD.CalcRDF),
        ("whim", rdMD.CalcWHIM),
        ("usr", rdMD.GetUSR),
        ("usrcat", rdMD.GetUSRCAT),
    ]:
        try:
            out[key] = _to_list(fn(mol))
        except Exception as exc:  # noqa: BLE001
            out[key] = None
            print(f"    {key}: {type(exc).__name__}: {exc}")
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

    # Idempotent: only compute compounds missing from the vector table.
    cur.execute("SELECT compound_id FROM compound_boltz2_desc3d_vector")
    already = {r[0] for r in cur.fetchall()}
    to_do = [(cid, p) for cid, p in rows if cid not in already]
    print(f"Already done: {len(already)}, to compute: {len(to_do)}")

    t0 = time.time()
    batch: list[tuple] = []
    failed: list[tuple[int, str]] = []
    for i, (cid, pkl_path) in enumerate(to_do):
        try:
            with open(pkl_path, "rb") as f:
                mol = pickle.load(f)
            if not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
                failed.append((cid, "no conformer"))
                continue
            d = compute(mol)
            batch.append(
                (
                    cid,
                    json.dumps(d["autocorr3d"]) if d["autocorr3d"] else None,
                    json.dumps(d["getaway"]) if d["getaway"] else None,
                    json.dumps(d["morse"]) if d["morse"] else None,
                    json.dumps(d["rdf"]) if d["rdf"] else None,
                    json.dumps(d["whim"]) if d["whim"] else None,
                    json.dumps(d["usr"]) if d["usr"] else None,
                    json.dumps(d["usrcat"]) if d["usrcat"] else None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((cid, f"{type(exc).__name__}: {exc}"))

        if len(batch) >= 250 or i + 1 == len(to_do):
            execute_batch(
                cur,
                """
                INSERT INTO compound_boltz2_desc3d_vector
                  (compound_id, autocorr3d, getaway, morse, rdf, whim, usr, usrcat)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (compound_id) DO UPDATE SET
                    autocorr3d = EXCLUDED.autocorr3d,
                    getaway    = EXCLUDED.getaway,
                    morse      = EXCLUDED.morse,
                    rdf        = EXCLUDED.rdf,
                    whim       = EXCLUDED.whim,
                    usr        = EXCLUDED.usr,
                    usrcat     = EXCLUDED.usrcat,
                    computed_at = NOW()
                """,
                batch,
            )
            conn.commit()
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(to_do) - (i + 1)) / max(rate, 1e-6)
            print(
                f"  [{i+1}/{len(to_do)}] flushed {len(batch)} rows  "
                f"rate={rate:.1f}/s  ETA={eta:.0f}s"
            )
            batch = []

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        for cid, reason in failed[:10]:
            print(f"  {cid}: {reason}")

    # Per-method non-null coverage
    cur.execute(
        """
        SELECT
          COUNT(*) AS total,
          COUNT(autocorr3d) AS n_autocorr3d,
          COUNT(getaway)    AS n_getaway,
          COUNT(morse)      AS n_morse,
          COUNT(rdf)        AS n_rdf,
          COUNT(whim)       AS n_whim,
          COUNT(usr)        AS n_usr,
          COUNT(usrcat)     AS n_usrcat
        FROM compound_boltz2_desc3d_vector
        """
    )
    cov = cur.fetchone()
    print(f"\nCoverage (non-null):")
    print(f"  total                {cov[0]}")
    for label, n in zip(
        ["autocorr3d", "getaway", "morse", "rdf", "whim", "usr", "usrcat"], cov[1:]
    ):
        print(f"  {label:<12} {n}")

    conn.close()


if __name__ == "__main__":
    main()
