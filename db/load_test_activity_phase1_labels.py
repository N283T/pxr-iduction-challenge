"""Load Phase 1 unblinded Track 1 test labels into PostgreSQL.

The table is intentionally separate from train_activity. Modeling code can then
choose when to include these labels for Phase 2 retraining without changing the
canonical Phase 1 training set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg2
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}
PARQUET_PATH = DATA_DIR.joinpath("phase_1_unblinded_test.parquet")


def _none_if_na(value):
    return None if pd.isna(value) else value


def load_phase1_frame() -> pd.DataFrame:
    """Return the Phase 1 unblinded test labels and cache them as parquet."""
    ds = load_dataset("openadmet/pxr-challenge-train-test", "phase_1_unblinded")
    df = ds["test"].to_pandas()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    return df


def main() -> None:
    df = load_phase1_frame()
    print(f"Loaded phase_1_unblinded/test from HF: {len(df)} rows")
    print(f"Cached parquet: {PARQUET_PATH}")

    conn = psycopg2.connect(**DB_PARAMS)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.smiles, c.id
            FROM compounds c
            JOIN test_activity t ON t.compound_id = c.id
            """
        )
        test_compound_map = {
            smiles: compound_id for smiles, compound_id in cur.fetchall()
        }

        missing = sorted(set(df["SMILES"]) - set(test_compound_map))
        if missing:
            raise RuntimeError(
                "Phase 1 unblinded rows missing from test_activity/compounds: "
                f"{len(missing)}"
            )

        for _, r in df.iterrows():
            cur.execute(
                """
                INSERT INTO test_activity_phase1_labels
                    (compound_id, phase, ocnt_batch, pec50,
                     pec50_ci_lower, pec50_ci_upper, pec50_std_error,
                     emax_estimate, emax_ci_lower, emax_ci_upper, emax_std_error,
                     emax_vs_pos_ctrl, emax_vs_pos_ctrl_ci_lower,
                     emax_vs_pos_ctrl_ci_upper, emax_vs_pos_ctrl_std_error,
                     source_split)
                VALUES
                    (%s, %s, %s, %s,
                     %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s)
                ON CONFLICT (compound_id, phase) DO UPDATE SET
                    ocnt_batch = EXCLUDED.ocnt_batch,
                    pec50 = EXCLUDED.pec50,
                    pec50_ci_lower = EXCLUDED.pec50_ci_lower,
                    pec50_ci_upper = EXCLUDED.pec50_ci_upper,
                    pec50_std_error = EXCLUDED.pec50_std_error,
                    emax_estimate = EXCLUDED.emax_estimate,
                    emax_ci_lower = EXCLUDED.emax_ci_lower,
                    emax_ci_upper = EXCLUDED.emax_ci_upper,
                    emax_std_error = EXCLUDED.emax_std_error,
                    emax_vs_pos_ctrl = EXCLUDED.emax_vs_pos_ctrl,
                    emax_vs_pos_ctrl_ci_lower = EXCLUDED.emax_vs_pos_ctrl_ci_lower,
                    emax_vs_pos_ctrl_ci_upper = EXCLUDED.emax_vs_pos_ctrl_ci_upper,
                    emax_vs_pos_ctrl_std_error = EXCLUDED.emax_vs_pos_ctrl_std_error,
                    source_split = EXCLUDED.source_split,
                    loaded_at = now()
                """,
                (
                    test_compound_map[r["SMILES"]],
                    int(r.get("phase", 1)),
                    r.get("OCNT Batch"),
                    r["pEC50"],
                    _none_if_na(r.get("pEC50_ci.lower (-log10(molarity))")),
                    _none_if_na(r.get("pEC50_ci.upper (-log10(molarity))")),
                    _none_if_na(r.get("pEC50_std.error (-log10(molarity))")),
                    _none_if_na(r.get("Emax_estimate (log2FC vs. baseline)")),
                    _none_if_na(r.get("Emax_ci.lower (log2FC vs. baseline)")),
                    _none_if_na(r.get("Emax_ci.upper (log2FC vs. baseline)")),
                    _none_if_na(r.get("Emax_std.error (log2FC vs. baseline)")),
                    _none_if_na(r.get("Emax.vs.pos.ctrl_estimate (dimensionless)")),
                    _none_if_na(r.get("Emax.vs.pos.ctrl_ci.lower (dimensionless)")),
                    _none_if_na(r.get("Emax.vs.pos.ctrl_ci.upper (dimensionless)")),
                    _none_if_na(r.get("Emax.vs.pos.ctrl_std.error (dimensionless)")),
                    r.get("Split"),
                ),
            )

        conn.commit()
        cur.execute("SELECT count(*) FROM test_activity_phase1_labels")
        total = cur.fetchone()[0]
        cur.execute(
            """
            SELECT count(*)
            FROM test_activity_phase1_labels p
            JOIN test_activity t ON t.compound_id = p.compound_id
            """
        )
        in_test = cur.fetchone()[0]
        print(f"Rows in test_activity_phase1_labels: {total}")
        print(f"Rows linked to test_activity: {in_test}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
