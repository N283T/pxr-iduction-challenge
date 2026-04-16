"""Compute all 217 RDKit 2D descriptors and store in compound_descriptors_full.

The existing `compound_descriptors` table only holds the ~41 descriptors exposed
by the PostgreSQL RDKit cartridge. This script populates a JSONB-backed table
with the full Descriptors._descList (BCUT2D, fr_*, VSA family, EState, MQN etc.)
computed via Python RDKit.
"""

import json
import logging
import math

import psycopg2
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}


def compute_descriptors(mol: Chem.Mol, desc_list: list) -> dict:
    """Compute all descriptors for a molecule, dropping non-finite values."""
    result: dict[str, float] = {}
    for name, fn in desc_list:
        try:
            val = fn(mol)
            fval = float(val)
        except Exception:
            continue
        if math.isfinite(fval):
            result[name] = fval
    return result


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS compound_descriptors_full (
            compound_id integer PRIMARY KEY REFERENCES compounds(id),
            descriptors jsonb NOT NULL
        )
        """
    )
    cur.execute("TRUNCATE compound_descriptors_full")
    conn.commit()

    cur.execute(
        "SELECT id, std_smiles FROM compounds WHERE std_mol IS NOT NULL ORDER BY id"
    )
    rows = cur.fetchall()
    logger.info("Computing full RDKit descriptors for %d compounds", len(rows))

    desc_list = list(Descriptors._descList)
    logger.info("RDKit 2D descriptor count: %d", len(desc_list))

    inserted = 0
    skipped = 0
    for cid, smiles in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue

        desc = compute_descriptors(mol, desc_list)
        cur.execute(
            "INSERT INTO compound_descriptors_full (compound_id, descriptors) VALUES (%s, %s)",
            (cid, json.dumps(desc)),
        )
        inserted += 1

        if inserted % 2000 == 0:
            conn.commit()
            logger.info("  Processed %d/%d", inserted, len(rows))

    conn.commit()

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_descriptors_full_compound "
        "ON compound_descriptors_full (compound_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_descriptors_full_descriptors "
        "ON compound_descriptors_full USING GIN (descriptors)"
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM compound_descriptors_full")
    total = cur.fetchone()[0]
    logger.info("Done. inserted=%d skipped=%d total_rows=%d", inserted, skipped, total)

    cur.execute(
        """
        SELECT compound_id,
               (SELECT COUNT(*) FROM jsonb_object_keys(descriptors))
        FROM compound_descriptors_full
        ORDER BY compound_id
        LIMIT 5
        """
    )
    for cid, nkeys in cur.fetchall():
        logger.info("  compound_id=%d -> %d descriptor keys", cid, nkeys)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
