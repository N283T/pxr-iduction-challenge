"""OOF -> LB gap analysis from `lb_submissions` + `experiment_summary`.

Known limitations:
  `lb_submissions` only holds submissions recorded after the api.py
  infrastructure was introduced. We have 3 rows at the time of writing
  (all 20-model UMAP ensembles from PR #46-#47). Historical submissions
  (ens_v6, ens_v7, etc.) were not tracked and their OOF<->LB pairs
  would need manual reconstruction from doc snapshots.

  With n=3 we cannot estimate a robust OOF->LB regression slope. We
  instead report the pairwise OOF/LB gap for all available pairs and
  note additional evidence from the ensemble cleanup doc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "eda_cv_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        df = pd.read_sql(
            """
            SELECT
                lb.submission_name,
                lb.experiment_name,
                es.rae_mean     AS oof_rae,
                es.mae_mean     AS oof_mae,
                lb.lb_rae,
                lb.lb_mae,
                lb.lb_rank,
                lb.notes,
                lb.submitted_at
            FROM lb_submissions lb
            LEFT JOIN experiment_summary es
                   ON lb.experiment_name = es.name
            ORDER BY lb.submitted_at
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("No lb_submissions rows found.")
        return

    df["gap_rae"] = df["lb_rae"] - df["oof_rae"]
    df["gap_mae"] = df["lb_mae"] - df["oof_mae"]
    df["ratio_rae"] = df["lb_rae"] / df["oof_rae"]

    print("=== Submission history with OOF/LB pairs ===")
    cols = [
        "submission_name",
        "oof_rae",
        "lb_rae",
        "gap_rae",
        "ratio_rae",
        "oof_mae",
        "lb_mae",
        "gap_mae",
        "lb_rank",
    ]
    print(df[cols].to_string(index=False))

    df.to_csv(OUT_DIR.joinpath("oof_lb_gap.csv"), index=False)

    print("\n=== Gap summary ===")
    print(f"n pairs: {len(df)}")
    print(
        f"RAE gap: mean={df['gap_rae'].mean():.4f}, "
        f"std={df['gap_rae'].std(ddof=0):.4f}, "
        f"min={df['gap_rae'].min():.4f}, max={df['gap_rae'].max():.4f}"
    )
    print(
        f"RAE ratio (LB/OOF): mean={df['ratio_rae'].mean():.4f}, "
        f"std={df['ratio_rae'].std(ddof=0):.4f}"
    )
    print(
        f"MAE gap: mean={df['gap_mae'].mean():.4f}, "
        f"std={df['gap_mae'].std(ddof=0):.4f}"
    )

    # Historical reference points from ensemble_cleanup.md:
    #   ens_v7_vanilla: OOF RAE 0.5304, LB RAE 0.62  (rank 14-17)
    # Adding those as context (not merged with the df above since there is
    # no DB row, and we do not want to silently mutate a dataset).
    print("\nHistorical reference from docs/ensemble_cleanup.md:")
    print("  ens_v7_vanilla: OOF RAE 0.5304 / LB RAE ~0.62  -> gap ~0.09")
    print("  (not in lb_submissions because submitted pre-api.py)")


if __name__ == "__main__":
    main()
