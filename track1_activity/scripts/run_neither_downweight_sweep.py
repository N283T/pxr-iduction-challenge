#!/usr/bin/env -S pixi run python
"""Sample-weight sweep for the 'neither' subset of train_activity.

Background (see docs/aux_revisit_observation.md):
    train splits into compounds that went through Octant's full assay flow
    ('both' subset, n=2,342, pec50 mean 4.85) and a separate diversity
    library that did not ('neither' subset, n=1,248, pec50 mean 3.33). The
    test set comes from analog expansion of potent hits and is
    distributionally closer to 'both'. This sweep down-weights the 'neither'
    subset during training to see whether the OOF improves.

Leakage safety:
    - The 'neither' assignment is only used to construct training-time
      sample_weights.
    - Features and inference are SMILES-derived only (Mordred).
    - All 513 test compounds are scored normally; sample_weight is not used
      at inference time.
    - This is the opposite of PR #41 which concatenated aux features into
      the model input.

Outputs:
    - stdout table of weight -> OOF metrics
    - No DB writes (this is a sweep, not a finished experiment)

Usage:
    pixi run python track1_activity/scripts/run_neither_downweight_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_mordred  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402

WEIGHTS_TO_TRY = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
N_SPLITS = 5
SEED = 42

# Default LightGBM params -- match the best baseline (`lgbm_mordred+morgan_r2`,
# OOF RAE ~0.553) so the sweep starts from a known reference point.
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
}


def load_train_with_neither_flag() -> pd.DataFrame:
    """Load train compounds with smiles, pec50, and is_neither boolean.

    is_neither = True iff compound has NO row in counter_assay AND NO row
    in single_concentration. This identifies the late-arrival diversity
    library compounds that bypassed the primary screen.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        """
        SELECT t.compound_id, c.std_smiles AS smiles, t.pec50,
               (ca.compound_id IS NULL) AS no_counter,
               (sc.compound_id IS NULL) AS no_single
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
        LEFT JOIN (
            SELECT DISTINCT compound_id FROM single_concentration
        ) sc ON sc.compound_id = t.compound_id
        ORDER BY t.id
        """,
        conn,
    )
    conn.close()
    df["is_neither"] = df["no_counter"] & df["no_single"]
    return df


def build_mordred_matrix(compound_ids: list[int]) -> np.ndarray:
    """Load Mordred descriptors aligned to compound_ids."""
    mordred_df = load_mordred(compound_ids)
    missing = set(compound_ids) - set(mordred_df.index)
    if missing:
        raise ValueError(f"Mordred missing for {len(missing)} compounds")
    X = mordred_df.loc[compound_ids].to_numpy(dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cv_oof_rae(
    X: np.ndarray,
    y: np.ndarray,
    is_neither: np.ndarray,
    splits: list,
    neither_weight: float,
) -> dict:
    """Run UMAP-fold CV and return OOF metrics for the given neither_weight."""
    oof_pred = np.zeros_like(y, dtype=np.float64)

    for tr_idx, va_idx in splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        w_tr = np.ones(len(tr_idx), dtype=np.float64)
        w_tr[is_neither[tr_idx]] = neither_weight

        dt = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)
        model = lgb.train(
            LGBM_PARAMS,
            dt,
            num_boost_round=2000,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        oof_pred[va_idx] = model.predict(X_va)

    metrics = compute_metrics(y, oof_pred)

    # Stratified by neither/both for diagnostic
    metrics_both = compute_metrics(y[~is_neither], oof_pred[~is_neither])
    metrics_neither = compute_metrics(y[is_neither], oof_pred[is_neither])

    return {
        "all": metrics,
        "both": metrics_both,
        "neither": metrics_neither,
    }


def main() -> None:
    print("Loading train data + Mordred...")
    df = load_train_with_neither_flag()
    n = len(df)
    n_neither = int(df["is_neither"].sum())
    print(f"  total: {n}, is_neither: {n_neither} ({n_neither / n:.1%})")
    print(
        f"  pec50 mean -- both: {df.loc[~df['is_neither'], 'pec50'].mean():.3f}, "
        f"neither: {df.loc[df['is_neither'], 'pec50'].mean():.3f}"
    )

    X = build_mordred_matrix(df["compound_id"].tolist())
    y = df["pec50"].to_numpy(dtype=np.float64)
    is_neither = df["is_neither"].to_numpy()
    print(f"  X shape: {X.shape}")

    print("\nBuilding UMAP folds...")
    splits = umap_split_indices(df["smiles"].tolist(), n_splits=N_SPLITS, seed=SEED)
    for i, (tr, va) in enumerate(splits):
        print(f"  fold {i}: train={len(tr)}, val={len(va)}")

    print("\nWeight sweep:")
    print(f"{'weight':>8} {'OOF RAE':>10} {'OOF MAE':>10} {'OOF R2':>9} "
          f"{'both RAE':>10} {'neither RAE':>13}")
    print("-" * 64)
    results = []
    for w in WEIGHTS_TO_TRY:
        m = cv_oof_rae(X, y, is_neither, splits, neither_weight=w)
        row = {
            "weight": w,
            "oof_rae": m["all"]["RAE"],
            "oof_mae": m["all"]["MAE"],
            "oof_r2": m["all"]["R2"],
            "both_rae": m["both"]["RAE"],
            "neither_rae": m["neither"]["RAE"],
        }
        results.append(row)
        print(f"{w:>8.2f} {row['oof_rae']:>10.4f} {row['oof_mae']:>10.4f} "
              f"{row['oof_r2']:>9.4f} {row['both_rae']:>10.4f} "
              f"{row['neither_rae']:>13.4f}")

    print("\nBaseline (weight=1.0) is the reference. Lower is better.")
    base = next(r for r in results if r["weight"] == 1.0)
    print("\nDelta vs baseline:")
    for r in results:
        delta = r["oof_rae"] - base["oof_rae"]
        marker = " <-- best" if r["oof_rae"] == min(x["oof_rae"] for x in results) else ""
        print(f"  weight={r['weight']:.2f}: ΔRAE={delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
