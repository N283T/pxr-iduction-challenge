"""Generate Boltz-2 input YAMLs for the 8,483 compounds NOT covered by the
main compound_boltz2 table.

Used for the "fast trunk" extension (Buterez strategy-3 on Boltz-2 trunk):
we only want trunk s/z embeddings at recycling_steps=1, no diffusion or
affinity. The paired wrapper (boltz2_fast_embeddings_run.sh) consumes the
YAMLs emitted here via boltz's ``--embeddings_only`` mode.

Track-1 placement (under ``track1_activity/scripts/boltz_affhead/``) — the
extension is a Track-1 feature engineering concern, not a Track-2 structure
run. Reuses ``track2_structure/src/boltz2/input_builder.py`` + constants for
YAML construction so the schema matches the existing 4,653-compound trunk.

Usage
-----
    pixi run python track1_activity/scripts/boltz_affhead/boltz2_build_inputs_fast.py
    pixi run python track1_activity/scripts/boltz_affhead/boltz2_build_inputs_fast.py --smoke  # 10 compounds only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]

# Reuse existing Boltz-2 YAML builder from Track 2
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from boltz2.input_builder import build_yaml, write_yaml  # noqa: E402
from data import DB_PARAMS  # noqa: E402

INPUTS_DIR = REPO_ROOT.joinpath("structures", "boltz2", "inputs_fast")
INPUTS_SMOKE_DIR = REPO_ROOT.joinpath("structures", "boltz2", "inputs_fast_smoke")
MANIFEST_PATH = REPO_ROOT.joinpath("structures", "boltz2", "manifest_fast.csv")
MANIFEST_SMOKE_PATH = REPO_ROOT.joinpath(
    "structures", "boltz2", "manifest_fast_smoke.csv"
)


MISSING_QUERY = """
    SELECT c.id, c.std_smiles
    FROM compounds c
    WHERE c.std_smiles IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM compound_boltz2 b WHERE b.compound_id = c.id
      )
    ORDER BY c.id
"""


def fetch_missing(limit: int | None = None) -> list[tuple[int, str]]:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        sql = (
            MISSING_QUERY
            if limit is None
            else f"{MISSING_QUERY}\n    LIMIT {int(limit)}"
        )
        cur.execute(sql)
        rows = cur.fetchall()
    return [(int(cid), str(smi)) for cid, smi in rows]


def write_inputs(
    compounds: list[tuple[int, str]],
    inputs_dir: Path,
    manifest_path: Path,
) -> None:
    """Write one YAML per compound + the manifest CSV.

    Uses ``build_yaml`` defaults: pocket_constraint=True, request_affinity=True.
    Affinity is skipped by ``--embeddings_only`` at inference; keeping the
    request in the YAML maintains schema parity with the existing 4,653-compound
    run so the file format is uniform.
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    written = 0
    for compound_id, smiles in compounds:
        yaml_path = inputs_dir.joinpath(f"{compound_id:05d}.yaml")
        if not yaml_path.exists():
            yaml_dict = build_yaml(smiles=smiles)
            write_yaml(yaml_dict, yaml_path)
            written += 1
        manifest_rows.append(
            {
                "compound_id": str(compound_id),
                "std_smiles": smiles,
                "yaml_path": str(yaml_path),
            }
        )

    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["compound_id", "std_smiles", "yaml_path"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[boltz2_build_inputs_fast] compounds queued: {len(manifest_rows)}")
    print(f"[boltz2_build_inputs_fast] new YAML files written: {written}")
    print(f"[boltz2_build_inputs_fast] inputs dir: {inputs_dir}")
    print(f"[boltz2_build_inputs_fast] manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: write 10 compounds to inputs_fast_smoke/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override compound count (default: 10 for smoke, all otherwise)",
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

    compounds = fetch_missing(limit=limit)
    if not compounds:
        raise SystemExit("No missing compounds found — nothing to emit.")

    write_inputs(
        compounds=compounds,
        inputs_dir=inputs_dir,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
