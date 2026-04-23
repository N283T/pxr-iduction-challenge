"""Export log2_fc training CSV for Uni-Mol MolTrain.

Produces two files under ``data/unimol/``:
  - pretrain_labeled.csv: compounds with at least one of log2fc_8p25/log2fc_33
    non-null (used for pretraining; NaN in other heads handled internally).
    Column schema: SMILES, log2fc_8p25, log2fc_33
  - pretrain_all.csv: all 13,136 compounds (compound_id + smiles only, for
    repr extraction after pretrain).
    Column schema: compound_id, SMILES

The ``SMILES`` column name (upper-case) is what unimol_tools expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "unimol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
SELECT c.id AS compound_id,
       c.std_smiles AS smiles,
       agg.log2fc_8p25,
       agg.log2fc_33
FROM compounds c
LEFT JOIN (
  SELECT compound_id,
    AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
             THEN log2_fc_estimate END) AS log2fc_8p25,
    AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
             THEN log2_fc_estimate END) AS log2fc_33
  FROM single_concentration
  GROUP BY compound_id
) agg ON agg.compound_id = c.id
WHERE c.std_smiles IS NOT NULL
ORDER BY c.id
"""


def main() -> None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(SQL, conn)

    n8 = df["log2fc_8p25"].notna().sum()
    n33 = df["log2fc_33"].notna().sum()
    print(f"Total: {len(df)}, log2fc_8p25 labeled: {n8}, log2fc_33 labeled: {n33}")

    labeled_mask = df["log2fc_8p25"].notna() | df["log2fc_33"].notna()
    df_lab = df[labeled_mask].rename(columns={"smiles": "SMILES"})[
        ["SMILES", "log2fc_8p25", "log2fc_33"]
    ]
    print(f"  labeled (at least one target): {len(df_lab)}")

    df_all = df.rename(columns={"smiles": "SMILES"})[["compound_id", "SMILES"]]
    print(f"  all (for repr): {len(df_all)}")

    df_lab.to_csv(OUT_DIR.joinpath("pretrain_labeled.csv"), index=False)
    df_all.to_csv(OUT_DIR.joinpath("pretrain_all.csv"), index=False)
    print(f"Wrote {OUT_DIR}/pretrain_labeled.csv ({len(df_lab)} rows)")
    print(f"Wrote {OUT_DIR}/pretrain_all.csv ({len(df_all)} rows)")


if __name__ == "__main__":
    main()
