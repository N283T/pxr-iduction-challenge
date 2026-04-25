"""Run PoseBusters pose quality checks on Boltz-2 predictions.

For every compound with a successful Boltz-2 prediction (i.e.
``compound_boltz2.preprocessing_failed = FALSE``), this script:

1. Extracts a protein-only PDB from the predicted mmCIF (Boltz cif
   includes both protein chain A and ligand chain B; PoseBusters wants
   a protein-only mol_cond).
2. Feeds the SDF pose (``structures/boltz2/ligands/<id>.sdf``) + the
   protein PDB to ``PoseBusters(config="dock")``.
3. Records the 19 boolean checks plus summary counts per compound.

Outputs a CSV and optionally upserts into
``compound_boltz2_posebusters``. Uses a process pool for parallelism
since each compound is independent and PoseBusters / RDKit release the
GIL enough for multiprocess scaling.

Usage
-----
    pixi run python track1_activity/boltz2/scripts/boltz2_posebusters.py --smoke
    pixi run python track1_activity/boltz2/scripts/boltz2_posebusters.py --db
    pixi run python track1_activity/boltz2/scripts/boltz2_posebusters.py --workers 8 --db
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import gemmi
import pandas as pd
import psycopg2
import psycopg2.extras

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.joinpath("src")))

from boltz2.constants import BOLTZ2_DIR, OUTPUTS_DIR, OUTPUTS_SMOKE_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Column mapping (PoseBusters raw name -> SQL-safe name)
# ---------------------------------------------------------------------------

COLUMN_MAP: dict[str, str] = {
    "mol_pred_loaded": "mol_pred_loaded",
    "mol_cond_loaded": "mol_cond_loaded",
    "sanitization": "sanitization",
    "inchi_convertible": "inchi_convertible",
    "all_atoms_connected": "all_atoms_connected",
    "no_radicals": "no_radicals",
    "bond_lengths": "bond_lengths",
    "bond_angles": "bond_angles",
    "internal_steric_clash": "internal_steric_clash",
    "aromatic_ring_flatness": "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness": "non_aromatic_ring_non_flatness",
    "double_bond_flatness": "double_bond_flatness",
    "internal_energy": "internal_energy",
    "protein-ligand_maximum_distance": "protein_ligand_maximum_distance",
    "minimum_distance_to_protein": "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors": "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors": "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters": "minimum_distance_to_waters",
    "volume_overlap_with_protein": "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors": "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors": "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters": "volume_overlap_with_waters",
}

INTRAMOL_COLS: frozenset[str] = frozenset(
    {
        "sanitization",
        "inchi_convertible",
        "all_atoms_connected",
        "no_radicals",
        "bond_lengths",
        "bond_angles",
        "internal_steric_clash",
        "aromatic_ring_flatness",
        "non_aromatic_ring_non_flatness",
        "double_bond_flatness",
        "internal_energy",
    }
)

INTERMOL_COLS: frozenset[str] = frozenset(
    {
        "protein_ligand_maximum_distance",
        "minimum_distance_to_protein",
        "minimum_distance_to_organic_cofactors",
        "minimum_distance_to_inorganic_cofactors",
        "minimum_distance_to_waters",
        "volume_overlap_with_protein",
        "volume_overlap_with_organic_cofactors",
        "volume_overlap_with_inorganic_cofactors",
        "volume_overlap_with_waters",
    }
)


# ---------------------------------------------------------------------------
# Protein-only PDB extraction
# ---------------------------------------------------------------------------


def write_protein_only_pdb(
    cif_path: Path, pdb_path: Path, ligand_chain: str = "B"
) -> None:
    """Strip ligand chain from a Boltz cif and write a protein-only PDB."""
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    for chain_name in [chain.name for chain in model]:
        if chain_name == ligand_chain:
            model.remove_chain(chain_name)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(pdb_path))


# ---------------------------------------------------------------------------
# PoseBusters per-compound worker (runs in a child process)
# ---------------------------------------------------------------------------


def _run_one(args: tuple[int, str, str, str]) -> dict[str, Any]:
    """Run PoseBusters on one compound. Called inside a worker process.

    Returns a dict with ``compound_id`` plus the SQL-mapped check columns
    and the summary counts. Missing files or PoseBusters exceptions are
    caught and the row is returned with ``error`` set.
    """
    compound_id, sdf_path_str, cif_path_str, pdb_path_str = args

    record: dict[str, Any] = {"compound_id": compound_id, "error": None}
    sdf_path = Path(sdf_path_str)
    cif_path = Path(cif_path_str)
    pdb_path = Path(pdb_path_str)

    if not sdf_path.exists() or not cif_path.exists():
        record["error"] = "missing sdf or cif"
        return record

    try:
        write_protein_only_pdb(cif_path, pdb_path)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"protein extraction failed: {type(exc).__name__}: {exc}"
        return record

    # Import inside worker to keep the parent process light.
    from posebusters import PoseBusters

    df = pd.DataFrame({"mol_pred": [str(sdf_path)], "mol_cond": [str(pdb_path)]})
    try:
        pb = PoseBusters(config="dock")
        result = pb.bust_table(df)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"posebusters failed: {type(exc).__name__}: {exc}"
        return record

    row = result.iloc[0]
    for raw_name, sql_name in COLUMN_MAP.items():
        value = row.get(raw_name)
        # Pandas NA -> None (non-applicable check, e.g. no cofactors)
        record[sql_name] = None if pd.isna(value) else bool(value)

    # Summary counts
    check_values = [record[sql_name] for sql_name in COLUMN_MAP.values()]
    passed = [v for v in check_values if v is True]
    record["num_checks"] = len([v for v in check_values if v is not None])
    record["num_passed"] = len(passed)
    record["all_passed"] = (
        record["num_checks"] > 0 and record["num_passed"] == record["num_checks"]
    )
    record["intramol_passed"] = all(record.get(col) is True for col in INTRAMOL_COLS)
    record["intermol_passed"] = all(
        record.get(col) in (True, None) for col in INTERMOL_COLS
    )

    return record


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def fetch_compound_file_paths(smoke: bool) -> list[tuple[int, str, str]]:
    """Return (compound_id, sdf_path, cif_path) for compounds to process.

    Picks only successful Boltz-2 predictions (pose_cif_path and
    ligand_sdf_path both non-null, preprocessing_failed = FALSE). In
    smoke mode, limits to the first 10 compound ids.
    """
    sql = """
        SELECT compound_id, ligand_sdf_path, pose_cif_path
        FROM compound_boltz2
        WHERE NOT preprocessing_failed
          AND ligand_sdf_path IS NOT NULL
          AND pose_cif_path IS NOT NULL
        ORDER BY compound_id
    """
    if smoke:
        sql += " LIMIT 10"
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [(int(cid), str(sdf), str(cif)) for cid, sdf, cif in cur.fetchall()]


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2].joinpath("db", "boltz2_posebusters_schema.sql")
)


def apply_schema() -> None:
    sql = SCHEMA_PATH.read_text()
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


DB_COLUMNS: tuple[str, ...] = (
    "compound_id",
    "num_checks",
    "num_passed",
    "all_passed",
    "intramol_passed",
    "intermol_passed",
    *COLUMN_MAP.values(),
    "posebusters_version",
)


def upsert_records(records: list[dict[str, Any]], pb_version: str) -> int:
    cols = list(DB_COLUMNS)
    placeholders = ", ".join(["%s"] * len(cols))
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "compound_id")
    sql = (
        f"INSERT INTO compound_boltz2_posebusters ({', '.join(cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (compound_id) DO UPDATE SET {update_clause}"
    )
    inserted = 0
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            for record in records:
                if record.get("error"):
                    continue  # skip errored compounds
                values = [record.get(c) for c in cols[:-1]] + [pb_version]
                cur.execute(sql, values)
                inserted += 1
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Process only 10 compounds."
    )
    parser.add_argument("--db", action="store_true", help="Upsert results into DB.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of worker processes (default: CPU count / 2)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Override results CSV output path.",
    )
    args = parser.parse_args()

    if args.smoke:
        out_csv = args.out_csv or BOLTZ2_DIR.joinpath("posebusters_smoke.csv")
        pdb_dir = OUTPUTS_SMOKE_DIR.joinpath("posebusters_tmp_proteins")
    else:
        out_csv = args.out_csv or BOLTZ2_DIR.joinpath("posebusters.csv")
        pdb_dir = OUTPUTS_DIR.joinpath("posebusters_tmp_proteins")
    pdb_dir.mkdir(parents=True, exist_ok=True)

    if args.db:
        apply_schema()
        print(f"[posebusters] applied schema: {SCHEMA_PATH.name}")

    inputs = fetch_compound_file_paths(smoke=args.smoke)
    print(f"[posebusters] compounds to process: {len(inputs)}")
    print(f"[posebusters] workers: {args.workers}")

    worker_args = [
        (cid, sdf, cif, str(pdb_dir.joinpath(f"{cid:05d}.pdb")))
        for cid, sdf, cif in inputs
    ]

    records: list[dict[str, Any]] = []
    if args.workers == 1:
        for wa in worker_args:
            records.append(_run_one(wa))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for i, rec in enumerate(
                pool.imap_unordered(_run_one, worker_args, chunksize=4)
            ):
                records.append(rec)
                if (i + 1) % 100 == 0:
                    print(f"[posebusters] processed {i + 1}/{len(worker_args)}")

    records.sort(key=lambda r: r["compound_id"])

    errors = [r for r in records if r.get("error")]
    successes = [r for r in records if not r.get("error")]
    print(f"[posebusters] success: {len(successes)}, error: {len(errors)}")

    # Write CSV
    if records:
        fieldnames = list(records[0].keys())
        # Make sure every record has all keys for CSV consistency.
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow({k: rec.get(k) for k in fieldnames})
        print(f"[posebusters] wrote CSV: {out_csv}")

    if args.db and successes:
        import posebusters

        pb_version = getattr(posebusters, "__version__", "unknown")
        inserted = upsert_records(successes, pb_version)
        print(f"[posebusters] upserted {inserted} rows")

    if errors:
        print("[posebusters] errored compounds:")
        for r in errors[:10]:
            print(f"  [{r['compound_id']}] {r['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
