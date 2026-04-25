"""Record permanently-failed Boltz-2 compounds in the compound_boltz2 table.

Two compounds could not be processed by Boltz-2 even after recovery
attempts and are flagged as permanently failed. They are recorded in
``compound_boltz2`` with ``preprocessing_failed=TRUE`` and a concise
``failure_reason`` so the downstream Track 1 ensemble can filter them
out explicitly rather than silently missing their rows.

The table itself is created by ``db/boltz2_schema.sql``; this script
applies the schema (idempotent via ``IF NOT EXISTS``) and then upserts
the two failed rows.

Details for each compound live in GitHub issue #50.
"""

from __future__ import annotations

from pathlib import Path

import psycopg2


SCHEMA_PATH = Path(__file__).resolve().parents[2].joinpath("db", "boltz2_schema.sql")

BOLTZ_VERSION = "2.2.1"

PERMANENTLY_FAILED = (
    {
        "compound_id": 1657,
        "failure_reason": (
            "Au-containing metal complex (Auranofin); excluded by "
            "Boltz-2 standardize. See issue #50."
        ),
    },
    {
        "compound_id": 3840,
        "failure_reason": (
            "RDKit ETKDGv3 3D embedding failed on (1S,4S)-2-aza-norbornane "
            "scaffold (PubChem CID 131950785). Not recoverable without "
            "external 3D source. See issue #50."
        ),
    },
)


UPSERT_SQL = """
    INSERT INTO compound_boltz2
        (compound_id, preprocessing_failed, failure_reason,
         ligand_oversize, boltz_version)
    VALUES (%s, TRUE, %s, FALSE, %s)
    ON CONFLICT (compound_id) DO UPDATE SET
        preprocessing_failed = EXCLUDED.preprocessing_failed,
        failure_reason       = EXCLUDED.failure_reason,
        ligand_oversize      = EXCLUDED.ligand_oversize,
        boltz_version        = EXCLUDED.boltz_version
"""


def main() -> None:
    schema_sql = SCHEMA_PATH.read_text()
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            print(f"[record-failures] applied schema from {SCHEMA_PATH}")

            for row in PERMANENTLY_FAILED:
                cur.execute(
                    UPSERT_SQL,
                    (row["compound_id"], row["failure_reason"], BOLTZ_VERSION),
                )
        conn.commit()
    print(
        f"[record-failures] upserted {len(PERMANENTLY_FAILED)} "
        "permanently-failed rows into compound_boltz2"
    )


if __name__ == "__main__":
    main()
