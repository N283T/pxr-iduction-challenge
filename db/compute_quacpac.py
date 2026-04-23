"""Compute OpenEye Quacpac pH 7.4 protonation + tautomer enumeration.

Writes two tables:
- ``compound_quacpac``: majority species at pH 7.4 (single SMILES + formal charge)
- ``compound_tautomers``: list of reasonable tautomers (JSONB SMILES array)

Requires ``OE_LICENSE=~/.openeye/oe_license.txt`` in the environment.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

N_WORKERS = 8
BATCH_SIZE = 500
MAX_TAUTOMERS = 16  # cap per compound to keep DB compact


def compute_one(task: tuple[int, str]) -> dict | None:
    compound_id, smiles = task
    from openeye import oechem, oequacpac

    # pH 7.4 protonation
    mol = oechem.OEGraphMol()
    if not oechem.OESmilesToMol(mol, smiles):
        return None
    try:
        oequacpac.OESetNeutralpHModel(mol)
        ph74_smiles = oechem.OEMolToSmiles(mol)
        formal_charge = int(oechem.OENetCharge(mol))
    except Exception:
        ph74_smiles = None
        formal_charge = None

    # Tautomer enumeration (from the *input* SMILES, not the protonated form)
    taut_mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(taut_mol, smiles)
    tautomers: list[str] = []
    try:
        for i, t in enumerate(oequacpac.OEGetReasonableTautomers(taut_mol)):
            if i >= MAX_TAUTOMERS:
                break
            tautomers.append(oechem.OEMolToSmiles(t))
    except Exception:
        pass

    return {
        "compound_id": compound_id,
        "ph74_smiles": ph74_smiles,
        "formal_charge": formal_charge,
        "input_smiles": smiles,
        "n_tautomers": len(tautomers),
        "tautomer_smiles": json.dumps(tautomers) if tautomers else None,
    }


def fetch_targets() -> list[tuple[int, str]]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.std_smiles
        FROM compounds c
        LEFT JOIN compound_quacpac q ON q.compound_id = c.id
        WHERE q.compound_id IS NULL AND c.std_smiles IS NOT NULL
        ORDER BY c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def upsert_batch(rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    # Split into two tables
    quacpac_cols = ("compound_id", "ph74_smiles", "formal_charge")
    quacpac_vals = [tuple(r[c] for c in quacpac_cols) for r in rows]
    execute_values(
        cur,
        """
        INSERT INTO compound_quacpac (compound_id, ph74_smiles, formal_charge)
        VALUES %s
        ON CONFLICT (compound_id) DO UPDATE SET
          ph74_smiles = EXCLUDED.ph74_smiles,
          formal_charge = EXCLUDED.formal_charge,
          computed_at = now()
        """,
        quacpac_vals,
    )
    taut_cols = ("compound_id", "input_smiles", "n_tautomers", "tautomer_smiles")
    taut_vals = [tuple(r[c] for c in taut_cols) for r in rows]
    execute_values(
        cur,
        """
        INSERT INTO compound_tautomers
            (compound_id, input_smiles, n_tautomers, tautomer_smiles)
        VALUES %s
        ON CONFLICT (compound_id) DO UPDATE SET
          input_smiles = EXCLUDED.input_smiles,
          n_tautomers = EXCLUDED.n_tautomers,
          tautomer_smiles = EXCLUDED.tautomer_smiles,
          computed_at = now()
        """,
        taut_vals,
    )
    conn.commit()
    cur.close()
    conn.close()


def main() -> None:
    if not os.environ.get("OE_LICENSE"):
        raise RuntimeError(
            "OE_LICENSE env not set. Run with "
            "OE_LICENSE=~/.openeye/oe_license.txt pixi run python ..."
        )

    targets = fetch_targets()
    print(f"Computing Quacpac pH 7.4 + tautomers for {len(targets)} compounds...")
    t0 = time.time()

    done = 0
    failed = 0
    with Pool(processes=N_WORKERS) as pool:
        buf: list[dict] = []
        for result in pool.imap_unordered(compute_one, targets, chunksize=50):
            if result is None:
                failed += 1
                continue
            buf.append(result)
            done += 1
            if len(buf) >= BATCH_SIZE:
                upsert_batch(buf)
                buf = []
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(targets) - done) / rate if rate > 0 else 0
                print(
                    f"  done={done}/{len(targets)} "
                    f"failed={failed} rate={rate:.1f}/s eta={eta:.0f}s"
                )
        upsert_batch(buf)

    elapsed = time.time() - t0
    print(f"\nFinished: done={done} failed={failed} elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
