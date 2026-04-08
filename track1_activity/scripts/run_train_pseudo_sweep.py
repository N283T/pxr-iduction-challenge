#!/usr/bin/env -S pixi run python
"""Sweep pseudo-label weights and report which (if any) improves baseline.

Iterates over pseudo-weights [0.05, 0.1, 0.3, 0.5, 1.0], skipping any run
whose experiment name already exists in the DB. After all runs, pulls the
6 relevant experiments (baseline + 5 pseudo weights) and prints a
comparison table sorted by OOF RAE ascending.

Usage:
    pixi run python track1_activity/scripts/run_train_pseudo_sweep.py
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data import DB_PARAMS  # noqa: E402
from run_train import run  # noqa: E402

PSEUDO_PATH = "data/pseudo_labels.parquet"
# Note: 0.3 may already be in DB from an earlier manual run; sweep dedupes by
# exp_name so it is safe to include here for reproducibility from a clean DB.
SWEEP_WEIGHTS = (0.05, 0.1, 0.3, 0.5, 1.0)
BASELINE_NAME = "lgbm_mordred_umap_default"
COMPARISON_NAMES = (
    BASELINE_NAME,
    "lgbm_mordred_umap_default_pseudo0.05",
    "lgbm_mordred_umap_default_pseudo0.1",
    "lgbm_mordred_umap_default_pseudo0.3",
    "lgbm_mordred_umap_default_pseudo0.5",
    "lgbm_mordred_umap_default_pseudo1.0",
)
SUMMARY_CSV = REPO_ROOT.joinpath(
    "track1_activity", "submissions", "pseudo_sweep_summary.csv"
)

OOF_RAE_RE = re.compile(r"OOF RAE=([0-9.]+)")


def build_args(weight: float) -> argparse.Namespace:
    return argparse.Namespace(
        model="lgbm",
        feature="mordred",
        split="umap",
        trials=0,
        gap_lambda=0.0,
        pseudo=PSEUDO_PATH,
        pseudo_weight=weight,
    )


def exp_name_for(weight: float) -> str:
    return f"{BASELINE_NAME}_pseudo{weight}"


def experiment_exists(name: str) -> bool:
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM experiments WHERE name = %s", (name,))
            return cur.fetchone() is not None


def fetch_comparison() -> pd.DataFrame:
    query = """
        SELECT e.id, e.name, e.notes,
               AVG(r.rae) AS fold_rae_mean,
               AVG(r.mae) AS fold_mae_mean
        FROM experiments e
        LEFT JOIN experiment_cv_results r ON r.experiment_id = e.id
        WHERE e.name = ANY(%s)
        GROUP BY e.id, e.name, e.notes
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(query, conn, params=(list(COMPARISON_NAMES),))

    def parse_oof(notes: str | None) -> float | None:
        if not notes:
            return None
        match = OOF_RAE_RE.search(notes)
        return float(match.group(1)) if match else None

    df = df.assign(oof_rae=df["notes"].map(parse_oof))
    baseline_row = df.loc[df["name"] == BASELINE_NAME]
    baseline_rae = (
        float(baseline_row["oof_rae"].iloc[0]) if not baseline_row.empty else None
    )
    df = df.assign(
        delta_vs_baseline=(
            df["oof_rae"] - baseline_rae if baseline_rae is not None else None
        )
    )
    return (
        df[["name", "oof_rae", "fold_mae_mean", "delta_vs_baseline", "id"]]
        .sort_values("oof_rae", na_position="last")
        .reset_index(drop=True)
    )


def main() -> None:
    print(f"Pseudo-weight sweep: {list(SWEEP_WEIGHTS)}")
    for weight in SWEEP_WEIGHTS:
        name = exp_name_for(weight)
        if experiment_exists(name):
            print(f"[skip] {name} already in DB")
            continue
        print(f"\n==== Running weight={weight} ({name}) ====")
        run(build_args(weight))

    print("\n==== Comparison ====")
    df = fetch_comparison()
    print(df.to_string(index=False))

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
