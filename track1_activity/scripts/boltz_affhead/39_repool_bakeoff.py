"""Bake off the raw-NPZ re-pooled Boltz trunk feature.

This script reports single-model OOF metrics and correlation to existing
Boltz trunk members after the model has been trained through run_train.py.
It does not modify ENSEMBLE_MODELS or submit anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402

DEFAULT_MODELS = (
    "lgbm_repooled_trunk_region_zstats_umap_default",
    "tabpfn_repooled_trunk_region_zstats_umap_default",
)
REFERENCES = (
    "tabpfn_pooled_boltz_allpairs_umap_default",
    "tabpfn_pooled_boltz_umap_default",
    "tabpfn_boltz_raw_plus_pretrain_concat_umap_default",
    "tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default",
)


def load_train_target() -> np.ndarray:
    with psycopg2.connect(**DB_PARAMS) as conn:
        y = pd.read_sql("SELECT pec50 FROM train_activity ORDER BY id", conn)
    return y["pec50"].to_numpy(dtype=np.float64)


def experiment_ids(names: tuple[str, ...]) -> dict[str, int]:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM experiments WHERE name = ANY(%s)",
            (list(names),),
        )
        return {name: int(exp_id) for exp_id, name in cur.fetchall()}


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    names = tuple(args.models) + REFERENCES
    ids = experiment_ids(names)
    missing = [n for n in names if n not in ids]
    if missing:
        print(f"Missing experiments: {missing}")

    y = load_train_target()
    oofs: dict[str, np.ndarray] = {}
    for name, exp_id in ids.items():
        oof = load_oof_predictions(exp_id)
        if oof is None or len(oof) != len(y):
            print(f"Skip {name}: invalid OOF")
            continue
        oofs[name] = oof

    rows = []
    for name in args.models:
        if name not in oofs:
            continue
        pred = oofs[name]
        row = {
            "name": name,
            "mae": mae(y, pred),
            "spearman": float(stats.spearmanr(y, pred).statistic),
        }
        for ref in REFERENCES:
            if ref in oofs:
                row[f"r_vs_{ref[:24]}"] = float(
                    stats.pearsonr(pred, oofs[ref]).statistic
                )
        rows.append(row)

    if not rows:
        raise SystemExit("No candidate OOFs available yet.")
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
