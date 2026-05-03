"""Export train_activity.pec50 only for direct pEC50 fine-tune.

Single target head. 4140 rows.

Output: data/unimol/pretrain_pec50.csv with columns SMILES, pec50.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "unimol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
SELECT c.std_smiles AS smiles, ta.pec50 AS pec50
FROM train_activity ta
JOIN compounds c ON c.id = ta.compound_id
WHERE c.std_smiles IS NOT NULL AND ta.pec50 IS NOT NULL
ORDER BY ta.compound_id
"""


def main() -> None:
    import psycopg2

    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(SQL, conn)
    df = df.rename(columns={"smiles": "SMILES"})
    print(f"pec50 train rows: {len(df)}")

    out = OUT_DIR.joinpath("pretrain_pec50.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
