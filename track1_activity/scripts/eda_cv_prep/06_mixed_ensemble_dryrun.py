"""Ensemble dry-run: does adding mixed-split OOFs to the current
UMAP-only candidate pool improve ensemble OOF MAE / RAE?

We do NOT modify run_ensemble.py. Instead we:
  1. Reuse its ENSEMBLE_MODELS allow list as the baseline pool.
  2. Load OOF predictions for each baseline model from the DB.
  3. Add the three new lgbm_*_mixed_default experiments to the pool.
  4. Compute L2-regularised convex weights with alpha=0.1 (matching
     the canonical ensemble strategy) for both pools.
  5. Report OOF MAE / RAE for (baseline) vs (baseline + mixed).

Note: the new mixed members are default-params (trials=0). If they
already help the ensemble at default quality, tuning will help more.
If they hurt, we should tune before concluding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import psycopg2
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402
from run_ensemble import ENSEMBLE_MODELS  # noqa: E402


MIXED_MODELS_DEFAULT = (
    "lgbm_mordred_jazzy_mixed_default",
    "lgbm_rdkit_desc_full_mixed_default",
    "lgbm_morgan_r2_2048_mixed_default",
)
MIXED_MODELS_TUNED = (
    "lgbm_mordred_jazzy_mixed",
    "lgbm_rdkit_desc_full_mixed",
    "lgbm_morgan_r2_2048_mixed",
)
MIXED_MODELS = MIXED_MODELS_TUNED


def load_y_train() -> np.ndarray:
    import pandas as pd

    conn = psycopg2.connect(**DB_PARAMS)
    try:
        df = pd.read_sql(
            "SELECT pec50 FROM train_activity ORDER BY id",
            conn,
        )
    finally:
        conn.close()
    return df["pec50"].to_numpy(dtype=np.float64)


def load_pool(names: list[str], y: np.ndarray) -> np.ndarray:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (names,),
    )
    rows = {r[1]: r[0] for r in cur.fetchall()}
    cur.close()
    conn.close()

    missing = [n for n in names if n not in rows]
    if missing:
        raise RuntimeError(f"Missing from DB: {missing}")

    oofs = []
    for n in names:
        oof = load_oof_predictions(rows[n])
        if oof is None:
            raise RuntimeError(f"{n}: no OOF in DB")
        if len(oof) != len(y):
            raise RuntimeError(f"{n}: OOF len {len(oof)} != y len {len(y)}")
        oofs.append(oof)
    return np.column_stack(oofs)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    wa = np.abs(w)
    total = wa.sum()
    if total < 1e-12:
        raise RuntimeError("all-zero weights")
    return wa / total


def _rae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def optimize_l2_rae(oof: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n

    def obj(w):
        wn = normalize_weights(w)
        return _rae(y, oof @ wn) + alpha * np.sum((wn - equal) ** 2)

    res = minimize(
        obj,
        equal.copy(),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    return normalize_weights(res.x)


def optimize_l2_mae(oof: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n

    def obj(w):
        wn = normalize_weights(w)
        return _mae(y, oof @ wn) + alpha * np.sum((wn - equal) ** 2)

    res = minimize(
        obj,
        equal.copy(),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    return normalize_weights(res.x)


def evaluate_pool(
    names: list[str],
    alpha: float = 0.1,
) -> dict:
    y = load_y_train()
    oof = load_pool(names, y)
    w_rae = optimize_l2_rae(oof, y, alpha)
    w_mae = optimize_l2_mae(oof, y, alpha)
    pred_rae = oof @ w_rae
    pred_mae = oof @ w_mae
    return {
        "n_models": len(names),
        "oof_rae_opt_rae": _rae(y, pred_rae),
        "oof_mae_opt_rae": _mae(y, pred_rae),
        "oof_rae_opt_mae": _rae(y, pred_mae),
        "oof_mae_opt_mae": _mae(y, pred_mae),
        "weights_rae": w_rae,
        "weights_mae": w_mae,
        "names": names,
    }


def main() -> None:
    print("Evaluating baseline (UMAP-only pool)...")
    baseline = evaluate_pool(list(ENSEMBLE_MODELS), alpha=0.1)
    print(f"  n={baseline['n_models']}")
    print(
        f"  opt=RAE: OOF RAE={baseline['oof_rae_opt_rae']:.4f}  "
        f"MAE={baseline['oof_mae_opt_rae']:.4f}"
    )
    print(
        f"  opt=MAE: OOF RAE={baseline['oof_rae_opt_mae']:.4f}  "
        f"MAE={baseline['oof_mae_opt_mae']:.4f}"
    )

    print("\nEvaluating baseline + mixed-default (3 new members)...")
    augmented_names = list(ENSEMBLE_MODELS) + list(MIXED_MODELS)
    augmented = evaluate_pool(augmented_names, alpha=0.1)
    print(f"  n={augmented['n_models']}")
    print(
        f"  opt=RAE: OOF RAE={augmented['oof_rae_opt_rae']:.4f}  "
        f"MAE={augmented['oof_mae_opt_rae']:.4f}"
    )
    print(
        f"  opt=MAE: OOF RAE={augmented['oof_rae_opt_mae']:.4f}  "
        f"MAE={augmented['oof_mae_opt_mae']:.4f}"
    )

    print("\n=== DELTA (augmented - baseline) ===")
    for k in (
        "oof_rae_opt_rae",
        "oof_mae_opt_rae",
        "oof_rae_opt_mae",
        "oof_mae_opt_mae",
    ):
        delta = augmented[k] - baseline[k]
        arrow = "[BETTER]" if delta < 0 else "[WORSE]"
        print(f"  {k}: {delta:+.4f}  {arrow}")

    # Weight assigned to mixed members (opt=MAE)
    mixed_start = len(ENSEMBLE_MODELS)
    print("\n=== Mixed members' weights (opt=MAE) ===")
    for name, w in zip(MIXED_MODELS, augmented["weights_mae"][mixed_start:]):
        print(f"  {name}: {w:.4f}")
    print(f"  sum of mixed weights: {augmented['weights_mae'][mixed_start:].sum():.4f}")


if __name__ == "__main__":
    main()
