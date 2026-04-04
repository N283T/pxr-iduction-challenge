"""Load parquet data into PostgreSQL."""

from pathlib import Path

import pandas as pd
import psycopg2

DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}


def get_conn():
    return psycopg2.connect(**DB_PARAMS)


def load_compounds(conn):
    """Insert unique SMILES from all datasets into compounds table."""
    train = pd.read_parquet(DATA_DIR.joinpath("default_train.parquet"))
    test = pd.read_parquet(DATA_DIR.joinpath("default_test.parquet"))
    counter = pd.read_parquet(DATA_DIR.joinpath("counter_assay_train.parquet"))
    single = pd.read_parquet(DATA_DIR.joinpath("single_concentration_train.parquet"))

    # Collect all unique (molecule_name, smiles) pairs
    all_compounds = pd.concat([
        train[["Molecule Name", "SMILES"]],
        test[["Molecule Name", "SMILES"]],
        counter[["Molecule Name", "SMILES"]],
        single[["Molecule Name", "SMILES"]].rename(columns={"Molecule Name": "Molecule Name"}),
    ]).drop_duplicates(subset=["SMILES"])

    print(f"Inserting {len(all_compounds)} unique compounds...")
    cur = conn.cursor()
    for _, row in all_compounds.iterrows():
        cur.execute(
            "INSERT INTO compounds (molecule_name, smiles) VALUES (%s, %s) ON CONFLICT (smiles) DO NOTHING",
            (row["Molecule Name"], row["SMILES"]),
        )
    conn.commit()
    cur.close()
    print(f"  Done. Compounds in DB: {_count(conn, 'compounds')}")


def _get_compound_id_map(conn):
    """Return {smiles: compound_id} mapping."""
    cur = conn.cursor()
    cur.execute("SELECT id, smiles FROM compounds")
    result = {smiles: cid for cid, smiles in cur.fetchall()}
    cur.close()
    return result


def _count(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
    n = cur.fetchone()[0]
    cur.close()
    return n


def load_train_activity(conn, compound_map):
    train = pd.read_parquet(DATA_DIR.joinpath("default_train.parquet"))
    print(f"Loading train_activity ({len(train)} rows)...")
    cur = conn.cursor()
    for _, r in train.iterrows():
        cur.execute(
            """INSERT INTO train_activity
            (compound_id, ocnt_batch, pec50, pec50_ci_lower, pec50_ci_upper, pec50_std_error,
             emax_estimate, emax_ci_lower, emax_ci_upper, emax_std_error,
             emax_vs_pos_ctrl, emax_vs_pos_ctrl_ci_lower, emax_vs_pos_ctrl_ci_upper, emax_vs_pos_ctrl_std_error)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                compound_map[r["SMILES"]],
                r.get("OCNT Batch"),
                r["pEC50"],
                r.get("pEC50_ci.lower (-log10(molarity))"),
                r.get("pEC50_ci.upper (-log10(molarity))"),
                r.get("pEC50_std.error (-log10(molarity))"),
                r.get("Emax_estimate (log2FC vs. baseline)"),
                r.get("Emax_ci.lower (log2FC vs. baseline)"),
                r.get("Emax_ci.upper (log2FC vs. baseline)"),
                r.get("Emax_std.error (log2FC vs. baseline)"),
                r.get("Emax.vs.pos.ctrl_estimate (dimensionless)"),
                r.get("Emax.vs.pos.ctrl_ci.lower (dimensionless)"),
                r.get("Emax.vs.pos.ctrl_ci.upper (dimensionless)"),
                r.get("Emax.vs.pos.ctrl_std.error (dimensionless)"),
            ),
        )
    conn.commit()
    cur.close()
    print(f"  Done. Rows: {_count(conn, 'train_activity')}")


def load_test_activity(conn, compound_map):
    test = pd.read_parquet(DATA_DIR.joinpath("default_test.parquet"))
    print(f"Loading test_activity ({len(test)} rows)...")
    cur = conn.cursor()
    for _, r in test.iterrows():
        cur.execute(
            "INSERT INTO test_activity (compound_id) VALUES (%s)",
            (compound_map[r["SMILES"]],),
        )
    conn.commit()
    cur.close()
    print(f"  Done. Rows: {_count(conn, 'test_activity')}")


def load_counter_assay(conn, compound_map):
    counter = pd.read_parquet(DATA_DIR.joinpath("counter_assay_train.parquet"))
    print(f"Loading counter_assay ({len(counter)} rows)...")
    cur = conn.cursor()
    for _, r in counter.iterrows():
        pec50 = r.get("pEC50")
        pec50 = None if pd.isna(pec50) else pec50
        cur.execute(
            """INSERT INTO counter_assay
            (compound_id, ocnt_batch, pec50, pec50_ci_lower, pec50_ci_upper, pec50_std_error,
             emax_estimate, emax_ci_lower, emax_ci_upper, emax_std_error,
             emax_vs_pos_ctrl, emax_vs_pos_ctrl_ci_lower, emax_vs_pos_ctrl_ci_upper, emax_vs_pos_ctrl_std_error)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                compound_map[r["SMILES"]],
                r.get("OCNT Batch"),
                pec50,
                None if pd.isna(v := r.get("pEC50_ci.lower (-log10(molarity))")) else v,
                None if pd.isna(v := r.get("pEC50_ci.upper (-log10(molarity))")) else v,
                None if pd.isna(v := r.get("pEC50_std.error (-log10(molarity))")) else v,
                None if pd.isna(v := r.get("Emax_estimate (log2FC vs. baseline)")) else v,
                None if pd.isna(v := r.get("Emax_ci.lower (log2FC vs. baseline)")) else v,
                None if pd.isna(v := r.get("Emax_ci.upper (log2FC vs. baseline)")) else v,
                None if pd.isna(v := r.get("Emax_std.error (log2FC vs. baseline)")) else v,
                None if pd.isna(v := r.get("Emax.vs.pos.ctrl_estimate (dimensionless)")) else v,
                None if pd.isna(v := r.get("Emax.vs.pos.ctrl_ci.lower (dimensionless)")) else v,
                None if pd.isna(v := r.get("Emax.vs.pos.ctrl_ci.upper (dimensionless)")) else v,
                None if pd.isna(v := r.get("Emax.vs.pos.ctrl_std.error (dimensionless)")) else v,
            ),
        )
    conn.commit()
    cur.close()
    print(f"  Done. Rows: {_count(conn, 'counter_assay')}")


def load_single_concentration(conn, compound_map):
    single = pd.read_parquet(DATA_DIR.joinpath("single_concentration_train.parquet"))
    print(f"Loading single_concentration ({len(single)} rows)...")
    cur = conn.cursor()
    for _, r in single.iterrows():
        cur.execute(
            """INSERT INTO single_concentration
            (compound_id, ocnt_batch, plate_id, compound_class, concentration_m,
             log2_fc_estimate, log2_fc_stderr, t_statistic, p_value, fdr_bh,
             neg_log10_fdr, median_log2_fc, cohens_d, n_replicates, experiment_name, ocnt_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                compound_map[r["SMILES"]],
                r.get("OCNT Batch"),
                r.get("plate_id"),
                r.get("compound_class"),
                r["concentration_M"],
                None if pd.isna(v := r.get("log2_fc_estimate")) else v,
                None if pd.isna(v := r.get("log2_fc_stderr")) else v,
                None if pd.isna(v := r.get("t_statistic")) else v,
                None if pd.isna(v := r.get("p_value")) else v,
                None if pd.isna(v := r.get("fdr_bh")) else v,
                None if pd.isna(v := r.get("neg_log10_fdr")) else v,
                None if pd.isna(v := r.get("median_log2_fc")) else v,
                None if pd.isna(v := r.get("cohens_d")) else v,
                None if pd.isna(v := r.get("n_replicates")) else int(v),
                r.get("experiment_name"),
                r.get("OCNT_ID"),
            ),
        )
    conn.commit()
    cur.close()
    print(f"  Done. Rows: {_count(conn, 'single_concentration')}")


def main():
    conn = get_conn()
    load_compounds(conn)
    compound_map = _get_compound_id_map(conn)
    load_train_activity(conn, compound_map)
    load_test_activity(conn, compound_map)
    load_counter_assay(conn, compound_map)
    load_single_concentration(conn, compound_map)
    conn.close()
    print("\nAll data loaded successfully.")


if __name__ == "__main__":
    main()
