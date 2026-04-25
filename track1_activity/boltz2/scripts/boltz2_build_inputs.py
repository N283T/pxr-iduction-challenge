"""Generate Boltz-2 input YAML files for the PXR train+test compounds.

Reads compounds from the pxr_challenge database, builds a YAML input per
compound, and writes a manifest CSV that maps compound_id to its YAML path
and the train/test split. The manifest is the only intermediate artifact
shared with the post-processing phase.

Usage
-----
    pixi run python track2_structure/scripts/boltz2_build_inputs.py             # full run (4653 compounds)
    pixi run python track2_structure/scripts/boltz2_build_inputs.py --smoke     # smoke test (10 compounds)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import psycopg2

# Make the boltz2 package importable when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.joinpath("src")))

from boltz2.constants import (  # noqa: E402
    INPUTS_DIR,
    INPUTS_SMOKE_DIR,
    MANIFEST_PATH,
    MANIFEST_SMOKE_PATH,
)
from boltz2.input_builder import build_yaml, write_yaml  # noqa: E402


COMPOUND_QUERY = """
    SELECT id, split, std_smiles FROM (
        SELECT DISTINCT c.id, 'train' AS split, c.std_smiles
        FROM compounds c
        JOIN train_activity t ON t.compound_id = c.id
        WHERE c.std_smiles IS NOT NULL
        UNION
        SELECT DISTINCT c.id, 'test' AS split, c.std_smiles
        FROM compounds c
        JOIN test_activity t ON t.compound_id = c.id
        WHERE c.std_smiles IS NOT NULL
    ) sub
    ORDER BY id
"""


def fetch_compounds(limit: int | None) -> list[tuple[int, str, str]]:
    """Fetch (compound_id, split, std_smiles) rows from the database."""
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            sql = COMPOUND_QUERY
            if limit is not None:
                sql = f"{sql} LIMIT {int(limit)}"
            cur.execute(sql)
            rows = cur.fetchall()
    return [(int(cid), str(split), str(smi)) for cid, split, smi in rows]


def write_inputs(
    compounds: list[tuple[int, str, str]],
    inputs_dir: Path,
    manifest_path: Path,
    use_pocket_constraint: bool,
) -> None:
    """Write a YAML per compound and the manifest CSV."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    written_count = 0
    for compound_id, split, smiles in compounds:
        yaml_path = inputs_dir.joinpath(f"{compound_id:05d}.yaml")
        if not yaml_path.exists():
            yaml_dict = build_yaml(
                smiles=smiles,
                use_pocket_constraint=use_pocket_constraint,
                request_affinity=True,
            )
            write_yaml(yaml_dict, yaml_path)
            written_count += 1
        manifest_rows.append(
            {
                "compound_id": str(compound_id),
                "split": split,
                "std_smiles": smiles,
                "yaml_path": str(yaml_path),
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["compound_id", "split", "std_smiles", "yaml_path"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[boltz2_build_inputs] compounds: {len(manifest_rows)}")
    print(f"[boltz2_build_inputs] new YAML files written: {written_count}")
    print(f"[boltz2_build_inputs] inputs dir: {inputs_dir}")
    print(f"[boltz2_build_inputs] manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: write 10 compounds to inputs_smoke/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override compound count (defaults: 10 for smoke, all otherwise).",
    )
    parser.add_argument(
        "--no-pocket",
        action="store_true",
        help="Disable the pocket constraint in the generated YAMLs.",
    )
    args = parser.parse_args()

    if args.smoke:
        inputs_dir = INPUTS_SMOKE_DIR
        manifest_path = MANIFEST_SMOKE_PATH
        limit = args.limit if args.limit is not None else 10
    else:
        inputs_dir = INPUTS_DIR
        manifest_path = MANIFEST_PATH
        limit = args.limit

    compounds = fetch_compounds(limit=limit)
    write_inputs(
        compounds=compounds,
        inputs_dir=inputs_dir,
        manifest_path=manifest_path,
        use_pocket_constraint=not args.no_pocket,
    )


if __name__ == "__main__":
    main()
