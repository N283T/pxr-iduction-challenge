"""Compute Jazzy hydration / H-bond descriptors and store in DB.

Jazzy (AstraZeneca, Sci Rep 2023) computes physically-grounded H-bond
donor/acceptor strengths and hydration free energies from 2D SMILES by
fitting a simple model on QM-derived references. The six features per
compound are:

    sdc   -- sum of donor strength, carbon H
    sdx   -- sum of donor strength, heteroatom H
    sa    -- sum of acceptor strength
    dga   -- apolar free energy of hydration (kJ/mol)
    dgp   -- polar free energy of hydration  (kJ/mol)
    dgtot -- total free energy of hydration  (kJ/mol)

These are orthogonal to Mordred's tpsa/logp approximations and were used
by several top teams in the ExpansionRx OpenADMET challenge (moka #3,
beetroot #9, crh201 #12, UncertainTea #19).

Output table: compound_jazzy (compound_id PK, six double precision cols).
"""

from __future__ import annotations

import logging
import sys
import warnings

import psycopg2
from jazzy.api import molecular_vector_from_smiles
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}

FEATURE_COLS = ("sdc", "sdx", "sa", "dga", "dgp", "dgtot")


def create_table(cur) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS compound_jazzy (
            compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
            {", ".join(f"{c} DOUBLE PRECISION" for c in FEATURE_COLS)}
        )
        """
    )


def compute_one(smiles: str) -> dict | None:
    try:
        return molecular_vector_from_smiles(smiles, minimisation_method="MMFF94")
    except Exception as e:  # noqa: BLE001 - jazzy raises many error types
        logger.debug("jazzy failed for %s: %s", smiles, e)
        return None


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    create_table(cur)
    cur.execute("TRUNCATE compound_jazzy")
    conn.commit()

    cur.execute(
        "SELECT id, std_smiles FROM compounds "
        "WHERE std_smiles IS NOT NULL ORDER BY id"
    )
    rows = cur.fetchall()
    logger.info("Computing Jazzy for %d compounds...", len(rows))

    insert_sql = (
        "INSERT INTO compound_jazzy "
        f"(compound_id, {', '.join(FEATURE_COLS)}) "
        f"VALUES (%s, {', '.join(['%s'] * len(FEATURE_COLS))})"
    )

    ok = 0
    fail = 0
    for i, (cid, smi) in enumerate(rows):
        vec = compute_one(smi)
        if vec is None:
            fail += 1
            continue
        try:
            values = (cid, *(float(vec[c]) for c in FEATURE_COLS))
        except (KeyError, TypeError, ValueError):
            fail += 1
            continue
        cur.execute(insert_sql, values)
        ok += 1

        if (i + 1) % 500 == 0:
            conn.commit()
            logger.info("  %d/%d (ok=%d, fail=%d)", i + 1, len(rows), ok, fail)

    conn.commit()
    logger.info("Done. ok=%d, fail=%d", ok, fail)

    if fail > 0:
        logger.warning("%d compounds failed Jazzy computation", fail)
        if fail > len(rows) * 0.05:
            logger.error("Failure rate >5%%, inspect input molecules")
            sys.exit(1)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_jazzy_compound ON compound_jazzy(compound_id)")
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
