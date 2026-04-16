"""Data loading utilities for Track 1."""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}
DB_URL = "postgresql+psycopg2:///pxr_challenge?host=/tmp&port=5433"

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


def get_engine():
    return create_engine(DB_URL)


def load_train_smiles_target():
    """Load train SMILES and pEC50."""
    return pd.read_sql(
        """SELECT c.std_smiles AS smiles, c.molecule_name, t.pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        get_engine(),
    )


def load_train_smiles_with_counter():
    """Load train SMILES, pEC50, and counter_assay pEC50 (NaN if missing).

    Same row order as load_train_smiles_target. The counter_pec50 column is
    NaN for train compounds that have no row in counter_assay (LEFT JOIN).
    """
    return pd.read_sql(
        """SELECT c.std_smiles AS smiles, c.molecule_name,
                  t.pec50, ca.pec50 AS counter_pec50
           FROM train_activity t
           JOIN compounds c ON c.id = t.compound_id
           LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
           ORDER BY t.id""",
        get_engine(),
    )


def load_train_smiles_multitask_5():
    """Load train SMILES + 5 task columns for multi-task learning.

    Columns:
        smiles, molecule_name, pec50 (main),
        counter_pec50, counter_emax,
        log2fc_8_25e_6, log2fc_3_30e_5

    NaN where the auxiliary measurement is missing. Same row order as
    load_train_smiles_target.
    """
    return pd.read_sql(
        """
        SELECT c.std_smiles AS smiles, c.molecule_name,
               t.pec50,
               ca.pec50 AS counter_pec50,
               ca.emax_estimate AS counter_emax,
               sc_hi.log2_fc_estimate AS log2fc_8_25e_6,
               sc_lo.log2_fc_estimate AS log2fc_3_30e_5
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
        LEFT JOIN single_concentration sc_hi
               ON sc_hi.compound_id = t.compound_id
              AND sc_hi.concentration_m BETWEEN 8.16e-6 AND 8.34e-6
        LEFT JOIN single_concentration sc_lo
               ON sc_lo.compound_id = t.compound_id
              AND sc_lo.concentration_m BETWEEN 3.27e-5 AND 3.33e-5
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_test_smiles():
    """Load test SMILES."""
    return pd.read_sql(
        """SELECT c.std_smiles AS smiles, c.molecule_name
           FROM test_activity t
           JOIN compounds c ON c.id = t.compound_id
           ORDER BY t.id""",
        get_engine(),
    )


def load_train_descriptors():
    """Load train data with descriptors."""
    desc = ", ".join(f"d.{c}" for c in DESCRIPTOR_COLS)
    return pd.read_sql(
        f"""SELECT c.std_smiles AS smiles, c.molecule_name, t.pec50, {desc}
            FROM train_activity t
            JOIN compounds c ON c.id = t.compound_id
            JOIN compound_descriptors d ON d.compound_id = c.id
            ORDER BY t.id""",
        get_engine(),
    )


def load_test_descriptors():
    """Load test data with descriptors."""
    desc = ", ".join(f"d.{c}" for c in DESCRIPTOR_COLS)
    return pd.read_sql(
        f"""SELECT c.std_smiles AS smiles, c.molecule_name, {desc}
            FROM test_activity t
            JOIN compounds c ON c.id = t.compound_id
            JOIN compound_descriptors d ON d.compound_id = c.id
            ORDER BY t.id""",
        get_engine(),
    )


def load_mordred(compound_ids: list[int] | None = None) -> pd.DataFrame:
    """Load Mordred descriptors from DB as a DataFrame.

    Returns DataFrame with compound_id as index, descriptor names as columns.
    """
    import json

    conn = get_conn()
    cur = conn.cursor()

    if compound_ids:
        placeholders = ",".join(["%s"] * len(compound_ids))
        cur.execute(
            f"SELECT compound_id, descriptors FROM compound_mordred WHERE compound_id IN ({placeholders})",
            compound_ids,
        )
    else:
        cur.execute(
            "SELECT compound_id, descriptors FROM compound_mordred ORDER BY compound_id"
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    records = []
    for compound_id, desc_json in rows:
        desc = desc_json if isinstance(desc_json, dict) else json.loads(desc_json)
        desc["compound_id"] = compound_id
        records.append(desc)

    return pd.DataFrame(records).set_index("compound_id")


def load_train_mordred() -> tuple[pd.DataFrame, pd.Series]:
    """Load train Mordred descriptors + pEC50."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id, pec50 FROM train_activity")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    compound_ids = [r[0] for r in rows]
    y = pd.Series({r[0]: r[1] for r in rows}, name="pec50")

    mordred_df = load_mordred(compound_ids)
    return mordred_df, y.loc[mordred_df.index]


def load_test_mordred() -> pd.DataFrame:
    """Load test Mordred descriptors."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM test_activity")
    compound_ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    return load_mordred(compound_ids)


def load_rdkit_full(compound_ids: list[int] | None = None) -> pd.DataFrame:
    """Load full 217-descriptor RDKit set from compound_descriptors_full.

    Returns DataFrame indexed by compound_id with descriptor names as columns.
    Missing descriptors (e.g. BCUT2D on metal-containing molecules) appear as
    NaN. Caller is responsible for imputation / dropping.
    """
    import json

    conn = get_conn()
    cur = conn.cursor()

    if compound_ids:
        placeholders = ",".join(["%s"] * len(compound_ids))
        cur.execute(
            "SELECT compound_id, descriptors FROM compound_descriptors_full "
            f"WHERE compound_id IN ({placeholders})",
            compound_ids,
        )
    else:
        cur.execute(
            "SELECT compound_id, descriptors FROM compound_descriptors_full "
            "ORDER BY compound_id"
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    records = []
    for cid, desc_json in rows:
        desc = desc_json if isinstance(desc_json, dict) else json.loads(desc_json)
        desc["compound_id"] = cid
        records.append(desc)

    return pd.DataFrame(records).set_index("compound_id")


def load_train_rdkit_full() -> tuple[pd.DataFrame, pd.Series]:
    """Load train full-RDKit descriptors + pEC50."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id, pec50 FROM train_activity")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    compound_ids = [r[0] for r in rows]
    y = pd.Series({r[0]: r[1] for r in rows}, name="pec50")

    desc_df = load_rdkit_full(compound_ids)
    return desc_df, y.loc[desc_df.index]


def load_test_rdkit_full() -> pd.DataFrame:
    """Load test full-RDKit descriptors."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM test_activity")
    compound_ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    return load_rdkit_full(compound_ids)


SINGLECONC_FEATURE_COLS = (
    "log2fc_8_25e_6",
    "log2fc_3_30e_5",
    "log2fc_stderr_8_25e_6",
    "cohens_d_8_25e_6",
    "p_value_8_25e_6",
    "n_concs",
)


def load_singleconc_features(compound_ids: list[int] | None = None) -> pd.DataFrame:
    """Load per-compound single_concentration features.

    Returns a DataFrame indexed by compound_id with the columns listed in
    SINGLECONC_FEATURE_COLS. Compounds without single_concentration data
    are returned with NaN across all columns. The two main concentrations
    8.25e-6 M and 3.30e-5 M are pivoted into separate columns.
    """
    eng = get_engine()
    where = ""
    params: dict = {}
    if compound_ids:
        placeholders = ",".join(str(int(i)) for i in compound_ids)
        where = f"WHERE compound_id IN ({placeholders})"

    sql = f"""
        SELECT compound_id, concentration_m, log2_fc_estimate,
               log2_fc_stderr, cohens_d, p_value
        FROM single_concentration {where}
    """
    raw = pd.read_sql(sql, eng)

    # Pivot to wide form per compound
    hi = (
        raw[raw["concentration_m"].between(8.16e-6, 8.34e-6)]
        .set_index("compound_id")[
            ["log2_fc_estimate", "log2_fc_stderr", "cohens_d", "p_value"]
        ]
        .rename(
            columns={
                "log2_fc_estimate": "log2fc_8_25e_6",
                "log2_fc_stderr": "log2fc_stderr_8_25e_6",
                "cohens_d": "cohens_d_8_25e_6",
                "p_value": "p_value_8_25e_6",
            }
        )
    )
    lo = (
        raw[raw["concentration_m"].between(3.27e-5, 3.33e-5)]
        .set_index("compound_id")[["log2_fc_estimate"]]
        .rename(columns={"log2_fc_estimate": "log2fc_3_30e_5"})
    )
    n_concs = raw.groupby("compound_id")["concentration_m"].nunique().rename("n_concs")

    feats = hi.join(lo, how="outer").join(n_concs, how="outer")

    if compound_ids:
        feats = feats.reindex(index=list(compound_ids))

    return feats[list(SINGLECONC_FEATURE_COLS)]


JAZZY_FEATURE_COLS = ("sdc", "sdx", "sa", "dga", "dgp", "dgtot")


def load_jazzy(compound_ids: list[int] | None = None) -> pd.DataFrame:
    """Load Jazzy hydration / H-bond features from DB.

    Returns a DataFrame indexed by compound_id with the six JAZZY_FEATURE_COLS.
    Compounds without a jazzy row are excluded (caller must align).
    """
    eng = get_engine()
    cols = ", ".join(JAZZY_FEATURE_COLS)
    if compound_ids:
        placeholders = ",".join(str(int(i)) for i in compound_ids)
        sql = (
            f"SELECT compound_id, {cols} FROM compound_jazzy "
            f"WHERE compound_id IN ({placeholders})"
        )
    else:
        sql = f"SELECT compound_id, {cols} FROM compound_jazzy ORDER BY compound_id"
    return pd.read_sql(sql, eng).set_index("compound_id")


def load_chemberta(compound_ids: list[int] | None = None) -> pd.DataFrame:
    """Load ChemBERTa embeddings from DB as a DataFrame.

    Returns DataFrame with compound_id as index, embedding dimensions as columns.
    """
    conn = get_conn()
    cur = conn.cursor()

    if compound_ids:
        placeholders = ",".join(["%s"] * len(compound_ids))
        cur.execute(
            f"SELECT compound_id, embedding FROM compound_chemberta WHERE compound_id IN ({placeholders})",
            compound_ids,
        )
    else:
        cur.execute(
            "SELECT compound_id, embedding FROM compound_chemberta ORDER BY compound_id"
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    data = {cid: emb for cid, emb in rows}
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "compound_id"
    df.columns = [f"chemberta_{i}" for i in range(df.shape[1])]
    return df


def load_train_chemberta() -> tuple[pd.DataFrame, pd.Series]:
    """Load train ChemBERTa embeddings + pEC50."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id, pec50 FROM train_activity")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    compound_ids = [r[0] for r in rows]
    y = pd.Series({r[0]: r[1] for r in rows}, name="pec50")

    emb_df = load_chemberta(compound_ids)
    return emb_df, y.loc[emb_df.index]


def load_test_chemberta() -> pd.DataFrame:
    """Load test ChemBERTa embeddings."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM test_activity")
    compound_ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    return load_chemberta(compound_ids)
