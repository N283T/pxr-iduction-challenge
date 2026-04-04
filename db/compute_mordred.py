"""Compute Mordred 2D descriptors and store in DB."""

import json
import logging

import numpy as np
import pandas as pd
import psycopg2
from mordred import Calculator, descriptors
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}


def main():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS compound_mordred (
            compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
            descriptors JSONB NOT NULL
        )
    """)
    cur.execute("TRUNCATE compound_mordred")
    conn.commit()

    # Fetch all compounds
    cur.execute("SELECT id, std_smiles FROM compounds WHERE std_mol IS NOT NULL ORDER BY id")
    compounds = cur.fetchall()
    logger.info(f"Computing Mordred descriptors for {len(compounds)} compounds...")

    calc = Calculator(descriptors, ignore_3D=True)
    descriptor_names = [str(d) for d in calc.descriptors]
    logger.info(f"Descriptor count: {len(descriptor_names)}")

    batch_size = 500
    total_inserted = 0

    for batch_start in range(0, len(compounds), batch_size):
        batch = compounds[batch_start : batch_start + batch_size]
        ids = [c[0] for c in batch]
        mols = [Chem.MolFromSmiles(c[1]) for c in batch]

        # Compute descriptors for batch
        result_df = calc.pandas(mols, quiet=True)

        for i, cid in enumerate(ids):
            row = result_df.iloc[i]
            desc_dict = {}
            for name, val in zip(descriptor_names, row):
                # Convert to native Python types, skip non-numeric
                try:
                    fval = float(val)
                    if np.isfinite(fval):
                        desc_dict[name] = fval
                except (ValueError, TypeError):
                    continue

            cur.execute(
                "INSERT INTO compound_mordred (compound_id, descriptors) VALUES (%s, %s)",
                (cid, json.dumps(desc_dict)),
            )

        conn.commit()
        total_inserted += len(batch)
        if total_inserted % 2000 == 0 or total_inserted == len(compounds):
            logger.info(f"  Processed {total_inserted}/{len(compounds)}")

    # Create GIN index for JSONB queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mordred_compound ON compound_mordred(compound_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mordred_descriptors ON compound_mordred USING gin(descriptors)")
    conn.commit()

    # Verify
    cur.execute("SELECT count(*) FROM compound_mordred")
    count = cur.fetchone()[0]
    logger.info(f"Done. {count} rows in compound_mordred")

    # Sample check
    cur.execute("""
        SELECT c.molecule_name, jsonb_object_keys(m.descriptors) AS key_count
        FROM compound_mordred m
        JOIN compounds c ON c.id = m.compound_id
        LIMIT 1
    """)
    cur.execute("""
        SELECT compound_id, jsonb_array_length(
            (SELECT jsonb_agg(k) FROM jsonb_object_keys(descriptors) k)
        ) AS num_keys
        FROM compound_mordred
        LIMIT 5
    """)
    for row in cur.fetchall():
        logger.info(f"  compound_id={row[0]}: {row[1]} descriptors stored")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
