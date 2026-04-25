"""Run Phase 2 post-processing for Boltz-2 outputs.

Walks the boltz output directory, builds a pose mol per compound (pkl
+ sdf), aggregates metadata into a CSV, and optionally upserts that
metadata into the ``compound_boltz2`` table. The CSV is always written
so the run is safe in dry-run mode.

Usage
-----
    pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py             # full set, csv only
    pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py --smoke     # 10-compound smoke set
    pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py --db        # csv + upsert into DB
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import psycopg2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.joinpath("src")))

from boltz2.constants import (  # noqa: E402
    BOLTZ2_DIR,
    INPUTS_DIR,
    INPUTS_SMOKE_DIR,
    MANIFEST_PATH,
    MANIFEST_SMOKE_PATH,
    OUTPUTS_DIR,
    OUTPUTS_SMOKE_DIR,
)
from boltz2.postprocess import collect_compound_metadata  # noqa: E402


# Column order matches db/boltz2_schema.sql (excluding boltz_version, created_at).
METADATA_COLUMNS: tuple[str, ...] = (
    "compound_id",
    "pose_cif_path",
    "ligand_pkl_path",
    "ligand_sdf_path",
    "confidence_json_path",
    "affinity_json_path",
    "plddt_npz_path",
    "pae_npz_path",
    "pde_npz_path",
    "embeddings_npz_path",
    "preprocessing_failed",
    "failure_reason",
    "ligand_oversize",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value_1",
    "affinity_probability_binary_1",
    "affinity_pred_value_2",
    "affinity_probability_binary_2",
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "ligand_atom_count",
    "ligand_to_pocket_distance_a",
)


def _resolve_results_subdir(out_dir: Path, input_dir: Path, subpath: str) -> Path:
    """Resolve ``<out_dir>/boltz_results_<input_name>/<subpath>``.

    Boltz writes into ``boltz_results_<basename>`` derived from the input
    directory's name, so we can compute the expected location instead of
    globbing. The glob-then-sorted-alphabetically fallback used previously
    silently routed post-processing to the wrong tree whenever more than
    one ``boltz_results_*`` directory was present (e.g. a leftover smoke
    directory next to the full run), which upserted failure rows for every
    compound. Require the expected directory; fall back to the legacy glob
    only when there is exactly one match and error out on ambiguity.
    """
    expected = out_dir.joinpath(f"boltz_results_{input_dir.name}", subpath)
    if expected.is_dir():
        return expected

    candidates = sorted(out_dir.glob(f"boltz_results_*/{subpath}"))
    if not candidates:
        raise FileNotFoundError(
            f"No {subpath} found under {out_dir}; expected {expected}."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple boltz_results_*/{subpath} directories under {out_dir} "
            f"but expected path {expected} is missing. Candidates: "
            f"{[str(c) for c in candidates]}"
        )
    return candidates[0]


def find_predictions_root(out_dir: Path, input_dir: Path) -> Path:
    """Locate the ``boltz_results_<input>/predictions`` subdirectory."""
    return _resolve_results_subdir(out_dir, input_dir, "predictions")


def find_mols_dir(out_dir: Path, input_dir: Path) -> Path:
    """Locate the ``boltz_results_<input>/processed/mols`` subdirectory."""
    return _resolve_results_subdir(out_dir, input_dir, "processed/mols")


def write_metadata_csv(records: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(METADATA_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col) for col in METADATA_COLUMNS})


def upsert_metadata_to_db(records: list[dict]) -> int:
    """Upsert metadata rows into ``compound_boltz2``. Returns the count."""
    if not records:
        return 0
    cols = list(METADATA_COLUMNS)
    placeholders = ", ".join(["%s"] * len(cols))
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "compound_id")
    sql = (
        f"INSERT INTO compound_boltz2 ({', '.join(cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (compound_id) DO UPDATE SET {update_clause}"
    )
    with psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge") as conn:
        with conn.cursor() as cur:
            for record in records:
                cur.execute(sql, [record.get(col) for col in cols])
        conn.commit()
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Operate on the 10-compound smoke set instead of the full run.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Upsert metadata into the compound_boltz2 table after writing the CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Override the metadata CSV output path.",
    )
    args = parser.parse_args()

    if args.smoke:
        input_dir = INPUTS_SMOKE_DIR
        out_dir = OUTPUTS_SMOKE_DIR
        manifest_path = MANIFEST_SMOKE_PATH
        pose_out_dir = BOLTZ2_DIR.joinpath("ligands_smoke")
        out_csv = args.out_csv or BOLTZ2_DIR.joinpath("postprocess_smoke.csv")
    else:
        input_dir = INPUTS_DIR
        out_dir = OUTPUTS_DIR
        manifest_path = MANIFEST_PATH
        pose_out_dir = BOLTZ2_DIR.joinpath("ligands")
        out_csv = args.out_csv or BOLTZ2_DIR.joinpath("postprocess.csv")

    predictions_root = find_predictions_root(out_dir, input_dir)
    mols_dir = find_mols_dir(out_dir, input_dir)
    print(f"[postprocess] predictions root: {predictions_root}")
    print(f"[postprocess] mols cache       : {mols_dir}")
    print(f"[postprocess] pose output dir  : {pose_out_dir}")

    manifest_rows: list[dict] = []
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            manifest_rows.append(row)
    print(f"[postprocess] manifest entries : {len(manifest_rows)}")

    records = []
    for row in manifest_rows:
        compound_id = int(row["compound_id"])
        record = collect_compound_metadata(
            predictions_dir=predictions_root,
            mols_dir=mols_dir,
            pose_out_dir=pose_out_dir,
            compound_id=compound_id,
        )
        records.append(record)

    write_metadata_csv(records, out_csv)
    print(f"[postprocess] wrote metadata CSV: {out_csv}")

    n_failed = sum(1 for r in records if r.get("preprocessing_failed"))
    n_oversize = sum(1 for r in records if r.get("ligand_oversize"))
    print(
        f"[postprocess] processed: {len(records)} "
        f"failed: {n_failed} oversize: {n_oversize}"
    )

    if args.db:
        inserted = upsert_metadata_to_db(records)
        print(f"[postprocess] upserted {inserted} rows into compound_boltz2")


if __name__ == "__main__":
    main()
