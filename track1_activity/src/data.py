"""Data loading utilities for Track 1."""

import psycopg2
import pandas as pd

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}

DESCRIPTOR_COLS = [
    "amw",
    "exactmw",
    "logp",
    "tpsa",
    "labuteasa",
    "fractioncsp3",
    "hba",
    "hbd",
    "num_atoms",
    "num_heavy_atoms",
    "num_heteroatoms",
    "num_rotatable_bonds",
    "num_amide_bonds",
    "num_rings",
    "num_aromatic_rings",
    "num_aliphatic_rings",
    "num_aromatic_carbocycles",
    "num_aromatic_heterocycles",
    "num_aliphatic_carbocycles",
    "num_aliphatic_heterocycles",
    "num_saturated_rings",
    "num_saturated_carbocycles",
    "num_saturated_heterocycles",
    "num_heterocycles",
    "num_spiro_atoms",
    "num_bridgehead_atoms",
    "chi0v",
    "chi1v",
    "chi2v",
    "chi3v",
    "chi4v",
    "chi0n",
    "chi1n",
    "chi2n",
    "chi3n",
    "chi4n",
    "kappa1",
    "kappa2",
    "kappa3",
    "phi",
    "hallkieralpha",
]


def get_conn():
    return psycopg2.connect(**DB_PARAMS)


def load_train_smiles_target():
    """Load train SMILES and pEC50."""
    conn = get_conn()
    df = pd.read_sql(
        """SELECT c.std_smiles AS smiles, c.molecule_name, t.pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id""",
        conn,
    )
    conn.close()
    return df


def load_test_smiles():
    """Load test SMILES."""
    conn = get_conn()
    df = pd.read_sql(
        """SELECT c.std_smiles AS smiles, c.molecule_name
           FROM test_activity t
           JOIN compounds c ON c.id = t.compound_id""",
        conn,
    )
    conn.close()
    return df


def load_train_descriptors():
    """Load train data with descriptors."""
    conn = get_conn()
    desc = ", ".join(f"d.{c}" for c in DESCRIPTOR_COLS)
    df = pd.read_sql(
        f"""SELECT c.std_smiles AS smiles, c.molecule_name, t.pec50, {desc}
            FROM train_activity t
            JOIN compounds c ON c.id = t.compound_id
            JOIN compound_descriptors d ON d.compound_id = c.id""",
        conn,
    )
    conn.close()
    return df


def load_test_descriptors():
    """Load test data with descriptors."""
    conn = get_conn()
    desc = ", ".join(f"d.{c}" for c in DESCRIPTOR_COLS)
    df = pd.read_sql(
        f"""SELECT c.std_smiles AS smiles, c.molecule_name, {desc}
            FROM test_activity t
            JOIN compounds c ON c.id = t.compound_id
            JOIN compound_descriptors d ON d.compound_id = c.id""",
        conn,
    )
    conn.close()
    return df
