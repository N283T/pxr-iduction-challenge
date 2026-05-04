#!/usr/bin/env -S pixi run python
"""MTR leak audit — gate G0.

Performs 6 leak-related checks on the MTR pretrain data setup.
Exits 0 with JSON report on success, exits 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports")
EXPECTED_NAN_COMPOUND_IDS = {1657, 8624}
EXPECTED_DESCRIPTOR_COUNT = 217


def check_id_overlap(conn) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT compound_id FROM train_activity"
        "  INTERSECT"
        "  SELECT compound_id FROM test_activity"
        ") AS x"
    )
    overlap = cur.fetchone()[0]
    return {"name": "L5a_compound_id_overlap", "passed": overlap == 0, "value": overlap}


def check_smiles_overlap(conn) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT c.std_smiles FROM compounds c JOIN train_activity t ON c.id = t.compound_id"
        "  INTERSECT"
        "  SELECT c.std_smiles FROM compounds c JOIN test_activity t ON c.id = t.compound_id"
        ") AS x"
    )
    overlap = cur.fetchone()[0]
    return {"name": "L5b_std_smiles_overlap", "passed": overlap == 0, "value": overlap}


def check_descriptor_source(conn) -> dict:
    # Read schema; verify only `descriptors` jsonb column exists for
    # compound_descriptors_full. No experiment-derived columns.
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'compound_descriptors_full' ORDER BY ordinal_position"
    )
    cols = [r[0] for r in cur.fetchall()]
    expected = ["compound_id", "descriptors"]
    return {
        "name": "L1_descriptor_source",
        "passed": cols == expected,
        "value": cols,
    }


def check_descriptor_count(conn) -> dict:
    df = pd.read_sql(
        "SELECT descriptors FROM compound_descriptors_full LIMIT 1",
        conn,
    )
    expanded = pd.json_normalize(df["descriptors"])
    n = len(expanded.columns)
    return {
        "name": "L6a_descriptor_count",
        "passed": n == EXPECTED_DESCRIPTOR_COUNT,
        "value": n,
    }


def check_nan_drop_set(conn) -> dict:
    df = pd.read_sql(
        "SELECT compound_id, descriptors FROM compound_descriptors_full",
        conn,
    )
    expanded = pd.json_normalize(df["descriptors"]).apply(
        pd.to_numeric, errors="coerce"
    )
    nan_rows = expanded.isna().any(axis=1)
    bad_ids = set(df.loc[nan_rows, "compound_id"].tolist())
    return {
        "name": "L6b_nan_drop_set",
        "passed": bad_ids == EXPECTED_NAN_COMPOUND_IDS,
        "value": sorted(bad_ids),
    }


def check_no_inf(conn) -> dict:
    df = pd.read_sql(
        "SELECT descriptors FROM compound_descriptors_full",
        conn,
    )
    expanded = pd.json_normalize(df["descriptors"]).apply(
        pd.to_numeric, errors="coerce"
    )
    inf_count = int(np.isinf(expanded.to_numpy()).sum())
    return {
        "name": "L6c_no_inf",
        "passed": inf_count == 0,
        "value": inf_count,
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with psycopg2.connect(**DB_PARAMS) as conn:
        checks = [
            check_id_overlap(conn),
            check_smiles_overlap(conn),
            check_descriptor_source(conn),
            check_descriptor_count(conn),
            check_nan_drop_set(conn),
            check_no_inf(conn),
        ]

    report = {
        "date": str(date.today()),
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }
    out_path = REPORT_DIR.joinpath(f"mtr_leak_audit_{date.today()}.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
