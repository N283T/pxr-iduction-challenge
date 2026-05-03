"""Export multi-task training CSV for Uni-Mol v2 multitask pretrain.

Targets (4 heads, NaN-safe per row):
  - log2fc_8p25: PXR cell line single-conc log2_fc at 8.25 µM (~10,875 cmpds)
  - log2fc_33:   PXR cell line single-conc log2_fc at 33 µM   (~10,875 cmpds)
  - pec50:       PXR dose-response pEC50                       (4,140 cmpds)
  - counter_pec50: PXR-null counter-assay pEC50                (2,860 cmpds)

Output: data/unimol/pretrain_multitask.csv with columns
  SMILES, log2fc_8p25, log2fc_33, pec50, counter_pec50

unimol_tools.MolTrain handles NaN per target via internal masking; rows with
NaN in head X just don't contribute to head X's loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2  # noqa: F401  (used inside data module)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "unimol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
SELECT c.id AS compound_id,
       c.std_smiles AS smiles,
       agg.log2fc_8p25,
       agg.log2fc_33,
       ta.pec50 AS pec50,
       ca.pec50 AS counter_pec50
FROM compounds c
LEFT JOIN (
  SELECT compound_id,
    AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
             THEN log2_fc_estimate END) AS log2fc_8p25,
    AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
             THEN log2_fc_estimate END) AS log2fc_33
  FROM single_concentration
  GROUP BY compound_id
) agg          ON agg.compound_id = c.id
LEFT JOIN train_activity ta ON ta.compound_id = c.id
LEFT JOIN counter_assay  ca ON ca.compound_id = c.id
WHERE c.std_smiles IS NOT NULL
ORDER BY c.id
"""


def main() -> None:
    import psycopg2 as pg

    with pg.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(SQL, conn)

    n8 = df["log2fc_8p25"].notna().sum()
    n33 = df["log2fc_33"].notna().sum()
    np_ = df["pec50"].notna().sum()
    nc = df["counter_pec50"].notna().sum()
    print(
        f"Total: {len(df)}, log2fc_8p25={n8}, log2fc_33={n33}, "
        f"pec50={np_}, counter_pec50={nc}"
    )

    # Subset: at least one of the 4 heads non-null
    mask = (
        df["log2fc_8p25"].notna()
        | df["log2fc_33"].notna()
        | df["pec50"].notna()
        | df["counter_pec50"].notna()
    )
    df_lab = df[mask].rename(columns={"smiles": "SMILES"})[
        ["SMILES", "log2fc_8p25", "log2fc_33", "pec50", "counter_pec50"]
    ]
    print(f"  rows with >= 1 label: {len(df_lab)}")

    out = OUT_DIR.joinpath("pretrain_multitask.csv")
    df_lab.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df_lab)} rows × 4 heads)")


if __name__ == "__main__":
    main()
