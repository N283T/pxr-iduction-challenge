"""Export compounds + log2_fc measurements to CSVs for KERMT pretrain.

Mirrors the SQL in run_chemprop_pretrain.py::load_pretrain_data.
Writes three CSVs:
  - pretrain_train.csv (90% random, seed=42): columns smiles, log2fc_8p25, log2fc_33
  - pretrain_val.csv (10%): same columns
  - pretrain_all.csv (all 13,136): columns smiles, compound_id
    (no labels; used for embed extraction only)

Usage:
    pixi run python track1_activity/scripts/prepare_kermt_pretrain_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "kermt")
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

    print(f"Loaded {len(df)} compounds")
    print(
        f"  log2fc_8p25 labeled: {df['log2fc_8p25'].notna().sum()}"
        f"  log2fc_33 labeled: {df['log2fc_33'].notna().sum()}"
    )

    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    df_train = df.iloc[train_idx][["smiles", "log2fc_8p25", "log2fc_33"]]
    df_val = df.iloc[val_idx][["smiles", "log2fc_8p25", "log2fc_33"]]
    df_all = df[["compound_id", "smiles"]]

    df_train.to_csv(OUT_DIR.joinpath("pretrain_train.csv"), index=False)
    df_val.to_csv(OUT_DIR.joinpath("pretrain_val.csv"), index=False)
    df_all.to_csv(OUT_DIR.joinpath("pretrain_all.csv"), index=False)

    print(f"Wrote {OUT_DIR.joinpath('pretrain_train.csv')} ({len(df_train)} rows)")
    print(f"Wrote {OUT_DIR.joinpath('pretrain_val.csv')} ({len(df_val)} rows)")
    print(f"Wrote {OUT_DIR.joinpath('pretrain_all.csv')} ({len(df_all)} rows)")


if __name__ == "__main__":
    main()
