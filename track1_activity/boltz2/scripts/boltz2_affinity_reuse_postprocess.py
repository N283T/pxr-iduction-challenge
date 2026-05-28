"""Load resumed Boltz-2 affinity outputs into a separate DB table.

The source run reuses saved official-Boltz trunk embeddings and writes
``affinity_<id>.json`` plus ``affinity_embeddings_<id>.npz``. Keep these rows
separate from ``compound_boltz2`` so downstream models can opt into the
revalidation run explicitly.

Usage
-----
    pixi run python track1_activity/boltz2/scripts/boltz2_affinity_reuse_postprocess.py
    pixi run python track1_activity/boltz2/scripts/boltz2_affinity_reuse_postprocess.py --db
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT.joinpath("db", "boltz2_affinity_reuse_schema.sql")
DEFAULT_PREDICTIONS_ROOT = REPO_ROOT.joinpath(
    "structures",
    "boltz2",
    "outputs_resume_from_embeddings",
    "boltz_results_inputs_affinity_reuse_all",
    "predictions",
)
DEFAULT_OUT_CSV = REPO_ROOT.joinpath(
    "structures", "boltz2", "affinity_reuse_postprocess.csv"
)

DB_PARAMS = {"host": "/tmp", "port": 5433, "dbname": "pxr_challenge"}

AFFINITY_KEYS = {
    "affinity_pred_value": "affinity_pred_value",
    "affinity_probability_binary": "affinity_probability_binary",
    "affinity_pred_value1": "affinity_pred_value_1",
    "affinity_probability_binary1": "affinity_probability_binary_1",
    "affinity_pred_value2": "affinity_pred_value_2",
    "affinity_probability_binary2": "affinity_probability_binary_2",
}

CSV_COLUMNS = (
    "compound_id",
    "affinity_json_path",
    "affinity_embeddings_npz_path",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value_1",
    "affinity_probability_binary_1",
    "affinity_pred_value_2",
    "affinity_probability_binary_2",
    "affinity_g1_dim",
    "affinity_g2_dim",
    "affinity_token_count",
)

DB_COLUMNS = (
    "compound_id",
    "affinity_json_path",
    "affinity_embeddings_npz_path",
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value_1",
    "affinity_probability_binary_1",
    "affinity_pred_value_2",
    "affinity_probability_binary_2",
    "affinity_g1",
    "affinity_g2",
    "affinity_token_count",
    "source_predictions_root",
    "boltz_version",
)


def repo_relative(path: Path) -> str:
    """Return a repository-relative path for stable DB storage."""
    return str(path.resolve().relative_to(REPO_ROOT))


def load_record(compound_dir: Path, predictions_root: Path) -> dict:
    """Load one compound's affinity JSON and final affinity embeddings."""
    cid = int(compound_dir.name)
    cid_str = f"{cid:05d}"
    json_path = compound_dir.joinpath(f"affinity_{cid_str}.json")
    emb_path = compound_dir.joinpath(f"affinity_embeddings_{cid_str}.npz")

    if not json_path.exists():
        raise FileNotFoundError(f"Missing affinity JSON: {json_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing affinity embeddings: {emb_path}")

    with json_path.open() as f:
        affinity = json.load(f)

    record = {
        "compound_id": cid,
        "affinity_json_path": repo_relative(json_path),
        "affinity_embeddings_npz_path": repo_relative(emb_path),
        "source_predictions_root": repo_relative(predictions_root),
    }
    for json_key, db_key in AFFINITY_KEYS.items():
        value = affinity.get(json_key)
        if value is None:
            raise ValueError(f"Missing {json_key} in {json_path}")
        record[db_key] = float(value)

    with np.load(emb_path) as data:
        for key in ("affinity_g1", "affinity_g2", "resume_token_idx"):
            if key not in data:
                raise ValueError(f"Missing {key} in {emb_path}")
        g1 = np.asarray(data["affinity_g1"], dtype=np.float64).reshape(-1)
        g2 = np.asarray(data["affinity_g2"], dtype=np.float64).reshape(-1)
        token_idx = np.asarray(data["resume_token_idx"])

    if g1.size != 384 or g2.size != 384:
        raise ValueError(
            f"Unexpected affinity embedding dims for {cid_str}: "
            f"g1={g1.size}, g2={g2.size}"
        )
    if token_idx.ndim != 1:
        raise ValueError(f"resume_token_idx must be 1D for {cid_str}")

    record["affinity_g1"] = g1.tolist()
    record["affinity_g2"] = g2.tolist()
    record["affinity_g1_dim"] = int(g1.size)
    record["affinity_g2_dim"] = int(g2.size)
    record["affinity_token_count"] = int(token_idx.size)
    return record


def collect_records(predictions_root: Path) -> list[dict]:
    """Collect all complete resumed-affinity prediction rows."""
    if not predictions_root.is_dir():
        raise FileNotFoundError(f"Predictions root not found: {predictions_root}")

    records = []
    for compound_dir in sorted(p for p in predictions_root.iterdir() if p.is_dir()):
        if not compound_dir.name.isdigit():
            continue
        records.append(load_record(compound_dir, predictions_root))
    return records


def write_csv(records: list[dict], out_csv: Path) -> None:
    """Write scalar metadata for quick inspection without dumping arrays."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col) for col in CSV_COLUMNS})


def apply_schema(conn) -> None:
    """Create the target table if needed."""
    with SCHEMA_PATH.open() as f:
        schema_sql = f.read()
    with conn.cursor() as cur:
        cur.execute(schema_sql)


def upsert_records(records: list[dict], boltz_version: str | None) -> int:
    """Upsert resumed affinity rows into compound_boltz2_affinity_reuse."""
    if not records:
        return 0

    update_clause = ", ".join(
        f"{col} = EXCLUDED.{col}" for col in DB_COLUMNS if col != "compound_id"
    )
    sql = (
        f"INSERT INTO compound_boltz2_affinity_reuse ({', '.join(DB_COLUMNS)}) "
        "VALUES %s "
        f"ON CONFLICT (compound_id) DO UPDATE SET {update_clause}"
    )
    rows = [
        tuple(
            record.get(col, boltz_version if col == "boltz_version" else None)
            for col in DB_COLUMNS
        )
        for record in records
    ]
    with psycopg2.connect(**DB_PARAMS) as conn:
        apply_schema(conn)
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=100)
        conn.commit()
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Directory containing per-compound affinity outputs.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Scalar metadata CSV path.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Create/update compound_boltz2_affinity_reuse.",
    )
    parser.add_argument(
        "--boltz-version",
        default="official-boltz codex/resume-embeddings-from-trunk",
        help="Provenance string stored in the DB.",
    )
    args = parser.parse_args()

    predictions_root = args.predictions_root.resolve()
    records = collect_records(predictions_root)
    write_csv(records, args.out_csv)

    print(f"[affinity_reuse] predictions root: {predictions_root}")
    print(f"[affinity_reuse] records         : {len(records)}")
    print(f"[affinity_reuse] wrote CSV       : {args.out_csv}")
    if records:
        token_counts = [record["affinity_token_count"] for record in records]
        print(
            "[affinity_reuse] token_count    : "
            f"min={min(token_counts)} max={max(token_counts)}"
        )

    if args.db:
        inserted = upsert_records(records, args.boltz_version)
        print(
            "[affinity_reuse] upserted       : "
            f"{inserted} rows into compound_boltz2_affinity_reuse"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
