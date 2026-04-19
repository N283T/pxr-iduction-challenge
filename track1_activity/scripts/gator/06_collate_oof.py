"""Collate per-fold Gator FT predictions into an OOF table + register
the experiment.

Inputs: structures/gator/ft_runs/<run_name>/fold{k}_val_preds.csv
        structures/gator/folds/fold{k}_val_cids.json

For each fold we align predictions with the compound_ids saved at
split time, then we look up train_idx (position in
`SELECT ... FROM train_activity ORDER BY id`) to match the
experiment_oof_predictions schema.

Compound 1657 (Auranofin) has no Boltz-2 pose and therefore no Gator
pose/pred. That one training row is omitted from OOF; the ensemble
layer is responsible for handling the missing value (either drop the
row from stacked OOF matrices, or impute with pool mean).

Options: --dry-run skips DB write and prints metrics; --register
inserts the experiment + OOF predictions table rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402


def load_fold_preds(run_dir: Path, k: int) -> dict[int, float]:
    csv_path = run_dir.joinpath(f"fold{k}", f"fold{k}_val_preds.csv")
    if not csv_path.exists():
        csv_path = run_dir.joinpath(f"fold{k}_val_preds.csv")
    df = pd.read_csv(csv_path)
    cid = df["PDB_ID"].astype(str).str.split("_").str[0].astype(int)
    return dict(zip(cid, df["Predicted_Affinity"].astype(float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_name under ft_runs/")
    ap.add_argument("--experiment-name", default="gator_pec50_ft_umap")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics and print; skip experiment/OOF DB writes.",
    )
    args = ap.parse_args()

    run_dir = REPO_ROOT.joinpath("structures", "gator", "ft_runs", args.run)
    print(f"Run dir: {run_dir}")

    cid_to_pred: dict[int, float] = {}
    for k in range(5):
        part = load_fold_preds(run_dir, k)
        dup = set(part) & set(cid_to_pred)
        if dup:
            raise RuntimeError(f"fold{k} duplicates: {sorted(dup)[:5]}...")
        cid_to_pred.update(part)
    print(f"OOF rows collated: {len(cid_to_pred)}")

    conn = psycopg2.connect(**DB_PARAMS)
    ta = pd.read_sql(
        "SELECT id, compound_id, pec50 FROM train_activity ORDER BY id", conn
    )
    ta["train_idx"] = np.arange(len(ta))

    # Build OOF vector aligned with train_idx ordering
    oof = np.full(len(ta), np.nan, dtype=np.float64)
    hits = 0
    for _, row in ta.iterrows():
        p = cid_to_pred.get(int(row["compound_id"]))
        if p is not None:
            oof[int(row["train_idx"])] = p
            hits += 1
    print(f"OOF rows mapped to train_idx: {hits}/{len(ta)}")

    mask = np.isfinite(oof)
    y = ta.loc[mask.nonzero()[0], "pec50"].to_numpy(dtype=np.float64)
    p = oof[mask]
    mae = float(np.mean(np.abs(y - p)))
    rae = float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    pr = stats.pearsonr(y, p)
    sp = stats.spearmanr(y, p)
    print()
    print(f"OOF metrics (on {int(mask.sum())} rows):")
    print(f"  MAE     {mae:.4f}")
    print(f"  RAE     {rae:.4f}")
    print(f"  RMSE    {rmse:.4f}")
    print(f"  Pearson {pr.statistic:+.4f}")
    print(f"  Spearman{sp.statistic:+.4f}")

    if args.dry_run:
        print("\n(dry run, not writing DB)")
        conn.close()
        return

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO experiments (name, model_type, feature_set, notes, cv_folds)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET
            notes  = EXCLUDED.notes
        RETURNING id
        """,
        (
            args.experiment_name,
            "gator_affinity",
            "gator_boltz2_pose_5A_pocket",
            f"5-fold UMAP FT (head-only) from run={args.run}; "
            f"OOF MAE={mae:.4f}, RAE={rae:.4f}",
            5,
        ),
    )
    exp_id = cur.fetchone()[0]
    print(f"\nexperiment_id = {exp_id}")

    # Write aggregate CV metrics into experiment_cv_results as a single
    # "fold=-1" summary row so experiment_summary pulls the same numbers.
    cur.execute(
        "DELETE FROM experiment_cv_results WHERE experiment_id = %s",
        (exp_id,),
    )
    cur.execute(
        """
        INSERT INTO experiment_cv_results
            (experiment_id, fold, mae, rae, spearman_r)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (exp_id, -1, mae, rae, sp.statistic),
    )

    # Clear existing OOF rows for this exp_id
    cur.execute(
        "DELETE FROM experiment_oof_predictions WHERE experiment_id = %s",
        (exp_id,),
    )
    rows = [(exp_id, int(idx), float(oof[idx])) for idx in np.where(mask)[0]]
    execute_batch(
        cur,
        "INSERT INTO experiment_oof_predictions (experiment_id, train_idx, oof_prediction) VALUES (%s, %s, %s)",
        rows,
        page_size=500,
    )
    conn.commit()
    print(f"Inserted {len(rows)} OOF rows")
    conn.close()


if __name__ == "__main__":
    main()
