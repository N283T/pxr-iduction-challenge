"""Caruana 2004 ensemble selection: forward stepwise selection with
replacement + sorted initialization + bagged 20x.

Motivation (see docs/papers/shotgun_raw/shotgun.txt):
  Our current Nelder-Mead optimization over normalized weights is
  continuous, unconstrained, and has a known failure mode: when two
  members are highly correlated, it greedily reallocates weight from
  the LB-stronger model to the OOF-look-alike challenger. Caruana
  selection uses discrete counts (replacement) and bagging over
  library subsets, which gives structural regularization against
  exactly this pattern.

Algorithm per bag:
  1. Sample a random fraction of the library (default 50%).
  2. Sort sampled members by single-model OOF MAE.
  3. Seed ensemble with the top `init_top_n` models (weight = 1 each).
  4. For `n_iter` iterations, pick the model whose addition minimises
     OOF MAE of (current_sum + cand_pred) / (current_count + 1).
     Model may already be in the ensemble (selection with replacement).
  5. Final bag ensemble = current_sum / current_count.
Bagged output = mean of `n_bags` bag ensembles.

All selection uses out-of-fold predictions (already unbiased by
construction of the 5-fold UMAP CV), so no separate hillclimb set.
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


EXISTING_8 = (
    "chemprop_optuna_umap",
    "chemprop_chemeleon_umap",
    "chemprop_multitask5_umap_aux0.0_tuned",
    "attentivefp_optuna_umap",
    "gatedgcn_optuna_umap",
    "residual_physprop+mordred_umap",
    "tabpfn_2d_full_boltz_umap",
    "tabpfn_chemeleon_umap",
)
NEW_3 = (
    "lgbm_pooled_boltz_umap",
    "tabpfn_pooled_boltz_umap_default",
    "tabpfn_pooled_boltz_allpairs_umap_default",
)
ALL_11 = EXISTING_8 + NEW_3


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rae(y, p):
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def caruana_select(
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    n_iter: int = 100,
    init_top_n: int = 5,
    bag_frac: float = 0.5,
    n_bags: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, int], list[dict]]:
    """Bagged Caruana selection. Returns (final_pred, counts, bag_info)."""
    rng = np.random.RandomState(seed)
    all_names = list(oofs.keys())
    N = len(y)
    bag_preds = []
    counts_total: dict[str, int] = {n: 0 for n in all_names}
    bag_info: list[dict] = []

    for b in range(n_bags):
        bag_size = max(init_top_n + 1, int(len(all_names) * bag_frac))
        bag_members = list(
            rng.choice(all_names, size=bag_size, replace=False)
        )
        bag_arr = np.stack([oofs[n] for n in bag_members], axis=0)  # (M, N)
        bag_maes = [mae(y, oofs[n]) for n in bag_members]
        sort_idx = np.argsort(bag_maes)
        top_n = sort_idx[:init_top_n]

        counts = np.zeros(len(bag_members), dtype=np.int64)
        counts[top_n] = 1
        current_sum = bag_arr[top_n].sum(axis=0)
        current_count = int(counts.sum())

        for it in range(n_iter):
            cand_pred = (
                current_sum[None, :] + bag_arr
            ) / (current_count + 1)
            cand_maes = np.mean(np.abs(cand_pred - y[None, :]), axis=1)
            best = int(np.argmin(cand_maes))
            counts[best] += 1
            current_sum = current_sum + bag_arr[best]
            current_count += 1

        bag_ens = current_sum / current_count
        bag_preds.append(bag_ens)
        bag_info.append(
            dict(
                bag=b,
                members=bag_members,
                counts=dict(zip(bag_members, counts.tolist())),
                bag_mae=mae(y, bag_ens),
            )
        )
        for name, cnt in zip(bag_members, counts):
            counts_total[name] += int(cnt)
        if verbose:
            print(
                f"  bag {b:>2}: members={len(bag_members)} "
                f"mae={mae(y, bag_ens):.4f}"
            )

    final_pred = np.mean(np.stack(bag_preds, axis=0), axis=0)
    return final_pred, counts_total, bag_info


def nelder_mead_weights(
    pool: list[str],
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    alpha: float = 0.0,
) -> tuple[np.ndarray, float]:
    X = np.column_stack([oofs[n] for n in pool])
    n = X.shape[1]
    eq = np.ones(n) / n

    def norm(w):
        a = np.abs(w)
        return a / a.sum()

    def obj(w):
        wn = norm(w)
        return mae(y, X @ wn) + alpha * np.sum((wn - eq) ** 2)

    r = minimize(
        obj,
        eq.copy(),
        method="Nelder-Mead",
        options=dict(maxiter=50000, xatol=1e-8, fatol=1e-8),
    )
    w = norm(r.x)
    return w, mae(y, X @ w)


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    y = pd.read_sql(
        "SELECT pec50 FROM train_activity ORDER BY id", conn
    )["pec50"].to_numpy(dtype=np.float64)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (list(ALL_11),),
    )
    ids = dict((n, i) for i, n in cur.fetchall())
    conn.close()
    missing = set(ALL_11) - set(ids)
    if missing:
        raise SystemExit(f"Missing: {missing}")

    oofs_11 = {n: load_oof_predictions(ids[n]) for n in ALL_11}
    oofs_8 = {n: oofs_11[n] for n in EXISTING_8}

    def report(label: str, pred: np.ndarray, pool: list[str], counts=None, weights=None):
        m = mae(y, pred)
        r = rae(y, pred)
        p = stats.pearsonr(y, pred).statistic
        print(f"\n=== {label} ===")
        print(f"  OOF MAE={m:.4f}  RAE={r:.4f}  Pearson={p:+.4f}")
        if counts is not None:
            total = sum(counts.values())
            print(f"  Counts (total={total}):")
            for name in sorted(counts, key=lambda n: -counts[n]):
                if counts[name] > 0:
                    w = counts[name] / total
                    print(f"    {w:>6.3f}  ({counts[name]:>3})  {name}")
        elif weights is not None:
            wd = dict(zip(pool, weights))
            for name in sorted(wd, key=lambda n: -wd[n]):
                if wd[name] > 0.01:
                    print(f"    {wd[name]:>6.3f}  {name}")

    # Baseline: Nelder-Mead on 8 and 11 (vanilla)
    w, m = nelder_mead_weights(list(EXISTING_8), oofs_11, y, alpha=0.0)
    report(
        "Nelder-Mead 8-base (vanilla)",
        np.column_stack([oofs_11[n] for n in EXISTING_8]) @ w,
        list(EXISTING_8),
        weights=w,
    )
    w, m = nelder_mead_weights(list(ALL_11), oofs_11, y, alpha=0.0)
    report(
        "Nelder-Mead 11 (vanilla)",
        np.column_stack([oofs_11[n] for n in ALL_11]) @ w,
        list(ALL_11),
        weights=w,
    )
    w, m = nelder_mead_weights(list(ALL_11), oofs_11, y, alpha=0.1)
    report(
        "Nelder-Mead 11 (L2 alpha=0.1)",
        np.column_stack([oofs_11[n] for n in ALL_11]) @ w,
        list(ALL_11),
        weights=w,
    )

    # Caruana variants
    print("\n" + "=" * 70)
    print("  CARUANA SELECTION")
    print("=" * 70)

    # 8-model pool, caruana
    pred, counts, _ = caruana_select(
        oofs_8, y, n_iter=100, init_top_n=3, bag_frac=0.5, n_bags=20, seed=42
    )
    report("Caruana 8 (init_top=3, iter=100, bags=20)", pred, list(EXISTING_8), counts=counts)

    # 11-model pool, varied settings
    for cfg in [
        dict(init_top_n=3, n_iter=100, bag_frac=0.5, n_bags=20),
        dict(init_top_n=5, n_iter=100, bag_frac=0.5, n_bags=20),
        dict(init_top_n=3, n_iter=100, bag_frac=0.7, n_bags=20),
        dict(init_top_n=0, n_iter=100, bag_frac=0.5, n_bags=20),  # no seed init
        dict(init_top_n=3, n_iter=200, bag_frac=0.5, n_bags=20),
        dict(init_top_n=3, n_iter=50, bag_frac=0.5, n_bags=20),
        dict(init_top_n=3, n_iter=100, bag_frac=1.0, n_bags=1),  # no bagging
    ]:
        tag = (
            f"Caruana 11 (init={cfg['init_top_n']}, iter={cfg['n_iter']}, "
            f"bag={cfg['bag_frac']}, bags={cfg['n_bags']})"
        )
        pred, counts, _ = caruana_select(oofs_11, y, seed=42, **cfg)
        report(tag, pred, list(ALL_11), counts=counts)


if __name__ == "__main__":
    main()
