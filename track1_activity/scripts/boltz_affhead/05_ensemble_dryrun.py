"""Ensemble dry-run: 8-model baseline vs +lgbm_pooled_boltz_umap.

Optimises weighted blend with the project's standard objective
(L2-penalised MAE, alpha=0.1) under canonical UMAP CV splits. Reports
OOF MAE/RAE/Pearson and the per-model weight for both pools so we can
see whether the new member pulls non-trivial weight and whether the
pooled-boltz signal displaces any existing member.

Mirrors the eda_cv_prep/14_prune_dryrun.py methodology.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import DB_PARAMS  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402


POOL_8 = (
    "chemprop_optuna_umap",
    "chemprop_chemeleon_umap",
    "chemprop_multitask5_umap_aux0.0_tuned",
    "attentivefp_optuna_umap",
    "gatedgcn_optuna_umap",
    "residual_physprop+mordred_umap",
    "tabpfn_2d_full_boltz_umap",
    "tabpfn_chemeleon_umap",
)

NEW_MEMBER = "lgbm_pooled_boltz_umap"


def normalize(w):
    a = np.abs(w)
    s = a.sum()
    if s < 1e-12:
        raise RuntimeError("weights collapsed")
    return a / s


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rae(y, p):
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def optimise(oof, y, alpha=0.1, metric="mae"):
    n = oof.shape[1]
    equal = np.ones(n) / n
    met_fn = mae if metric == "mae" else rae

    def obj(w):
        wn = normalize(w)
        return met_fn(y, oof @ wn) + alpha * np.sum((wn - equal) ** 2)

    r = minimize(
        obj,
        equal.copy(),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return normalize(r.x)


def evaluate(pool, y, ids):
    oofs = np.column_stack([load_oof_predictions(ids[n]) for n in pool])
    w = optimise(oofs, y, alpha=0.1, metric="mae")
    pred = oofs @ w
    return dict(
        n=len(pool),
        mae=mae(y, pred),
        rae=rae(y, pred),
        pearson=float(stats.pearsonr(y, pred).statistic),
        weights=dict(zip(pool, w)),
    )


def main():
    conn = psycopg2.connect(**DB_PARAMS)
    y = pd.read_sql(
        "SELECT pec50 FROM train_activity ORDER BY id", conn
    )["pec50"].to_numpy(dtype=np.float64)
    cur = conn.cursor()
    all_names = set(POOL_8) | {NEW_MEMBER}
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (list(all_names),),
    )
    ids = dict((n, i) for i, n in cur.fetchall())
    conn.close()
    missing = all_names - set(ids)
    if missing:
        raise SystemExit(f"Missing: {missing}")

    print(f"{'pool':<38} {'n':>3} {'MAE':>7} {'RAE':>7} {'Pearson':>9}")
    print("-" * 68)
    r8 = evaluate(POOL_8, y, ids)
    print(
        f"{'A: 8-model baseline':<38} {r8['n']:>3} "
        f"{r8['mae']:>7.4f} {r8['rae']:>7.4f} {r8['pearson']:>+9.4f}"
    )
    r9 = evaluate(POOL_8 + (NEW_MEMBER,), y, ids)
    print(
        f"{'B: +lgbm_pooled_boltz_umap':<38} {r9['n']:>3} "
        f"{r9['mae']:>7.4f} {r9['rae']:>7.4f} {r9['pearson']:>+9.4f}"
    )
    print(
        f"\n  Δ(B - A) = MAE {r9['mae'] - r8['mae']:+.4f}  "
        f"RAE {r9['rae'] - r8['rae']:+.4f}"
    )

    # Detailed weight comparison
    print("\n=== Weights (descending) ===")
    print(f"{'model':<42} {'A (8-model)':>12} {'B (+new)':>10}")
    print("-" * 68)
    ordered = sorted(
        POOL_8 + (NEW_MEMBER,),
        key=lambda n: -r9["weights"].get(n, 0),
    )
    for n in ordered:
        wA = r8["weights"].get(n, None)
        wB = r9["weights"].get(n, None)
        sA = f"{wA:.3f}" if wA is not None else "   -  "
        sB = f"{wB:.3f}" if wB is not None else "   -  "
        marker = "  **" if n == NEW_MEMBER else ""
        print(f"{n:<42} {sA:>12} {sB:>10}{marker}")

    # Also sanity: try drop-one replacement (swap tabpfn_2d_full_boltz with new,
    # since that member is 0.91 correlated and would be the candidate to cede
    # weight).
    swap_pool = tuple(
        NEW_MEMBER if m == "tabpfn_2d_full_boltz_umap" else m
        for m in POOL_8
    )
    rs = evaluate(swap_pool, y, ids)
    print(
        f"\nC: swap tabpfn_2d_full_boltz -> new  "
        f"MAE={rs['mae']:.4f}  RAE={rs['rae']:.4f}  "
        f"Pearson={rs['pearson']:+.4f}  "
        f"Δ(C-A)={rs['mae']-r8['mae']:+.4f}"
    )


if __name__ == "__main__":
    main()
