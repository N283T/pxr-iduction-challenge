"""Counter-assay decomposition (issue #113 Phase 1).

Hypothesis (Codex 2026-04-23): the PXR pEC50 signal can be split as
``y = c + s`` where ``c`` is a counter-assay-driven off-target component
and ``s`` is a PXR-selective residual. Two separate regressors fit
``c_hat`` and ``s_hat``, and we predict ``y_hat = c_hat + s_hat``.

Why try this after stacker null:
  - Stacker is bounded by the 9-pool OOF subspace; it averages signal
    pool members already extract.
  - Counter assay (compound_id -> off-target pEC50) is information no
    pool member explicitly uses for its prediction. 0/513 test compounds
    have counter labels but the model can still extrapolate.
  - 2648/4140 train compounds have usable counter_pec50 (2648 not-NULL,
    out of 2860 total counter rows; 212 NULL labels).
  - Selectivity (y - c) median 1.71, mean 1.61 ± 1.35.

Cross-fit handling:
  - Outer 5-fold UMAP (canonical seed=42, k=50, Morgan+Jaccard).
  - For each outer fold, M_c is trained on counter-labelled samples in
    fold_train (~80% of 2648 = ~2118). c_hat for THOSE counter samples
    is generated via inner 5-fold ``cross_val_predict`` so the residual
    s = y - c_hat used to fit M_s is leak-free.
  - For non-counter-labelled fold_train compounds, c_hat comes from a
    full-fit M_c on (counter ∩ fold_train) — no leak since they have no
    counter label that could leak.
  - For fold_val compounds, c_hat comes from the same full-fit M_c.
  - M_s is then fit on (X_fold_train, y_fold_train - c_hat_fold_train).

Decision gates:
  - Phase 1A: y_hat OOF MAE < direct M_total OOF MAE by >= 0.005 AND
    < caruana_bag20 0.4150 by >= 0.003. Same 217d rdkit_desc_full
    feature set + LGBM defaults for both.
  - Phase 1B (only if 1A passes): scale to cheme_2d_full_boltz_log2fc_pred
    (2103d) + TabPFN, evaluate against pool weakest 0.486 single-MAE
    gate and caruana_bag20 ADD gate.

Usage:
    pixi run python track1_activity/scripts/run_counter_decomp.py
    pixi run python track1_activity/scripts/run_counter_decomp.py --feature mordred --inner-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import (  # noqa: E402
    DB_PARAMS,
    load_rdkit_full,
    load_mordred,
    load_train_smiles_target,
)
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_train_compound_ids() -> list[int]:
    """Compound IDs in train_activity, ordered by train_activity.id.

    Matches the ordering convention of load_train_smiles_target so X / y
    rows are perfectly aligned across all loaders in this script.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    cids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return cids


def _impute(X: np.ndarray) -> np.ndarray:
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col[mask] = float(np.nanmedian(col))
            X[:, j] = col
    return X


def load_features(feature: str) -> tuple[np.ndarray, list[int]]:
    """Return (X_train, train_compound_ids ordered by train_activity.id)."""
    cids = _load_train_compound_ids()

    if feature == "rdkit_desc_full":
        df = load_rdkit_full(cids)  # indexed by compound_id, sorted by id
        df = df.reindex(cids)  # align to train_activity.id order
        return _impute(df.to_numpy(dtype=np.float64)), cids
    if feature == "mordred":
        df = load_mordred(cids)
        df = df.reindex(cids)
        return _impute(df.to_numpy(dtype=np.float64)), cids
    raise ValueError(f"Unknown feature: {feature}")


def load_counter_for_train(train_compound_ids: list[int]) -> np.ndarray:
    """Return counter_pec50 vector aligned to ``train_compound_ids``,
    NaN where counter_pec50 is NULL or absent.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        """SELECT compound_id, pec50 AS counter_pec50
           FROM counter_assay
           WHERE compound_id = ANY(%(ids)s)""",
        conn,
        params={"ids": train_compound_ids},
    )
    conn.close()
    cmap = dict(zip(df["compound_id"], df["counter_pec50"]))
    return np.array(
        [
            cmap.get(cid) if cmap.get(cid) is not None else np.nan
            for cid in train_compound_ids
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def fit_lgbm(X: np.ndarray, y: np.ndarray, *, seed: int = 42, num_round: int = 500):
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        "seed": seed,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.01,
        "lambda_l2": 1.0,
    }
    dtrain = lgb.Dataset(X, label=y)
    return lgb.train(params, dtrain, num_boost_round=num_round)


def cross_val_predict_lgbm(
    X: np.ndarray, y: np.ndarray, *, seed: int = 42, n_splits: int = 5
) -> np.ndarray:
    """Stratification-free K-fold OOF predictions via LGBM."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full_like(y, np.nan, dtype=np.float64)
    for tr_idx, va_idx in kf.split(X):
        m = fit_lgbm(X[tr_idx], y[tr_idx], seed=seed)
        oof[va_idx] = m.predict(X[va_idx])
    return oof


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    print(f"Loading features: {args.feature}")
    X, cid_order = load_features(args.feature)
    print(f"  X shape {X.shape}")

    train_df = load_train_smiles_target()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    smiles = train_df["smiles"].tolist()
    if len(y) != X.shape[0]:
        raise RuntimeError(f"X rows {X.shape[0]} != y rows {len(y)}")

    # counter labels aligned to row order
    c_label = load_counter_for_train(cid_order)
    has_counter = ~np.isnan(c_label)
    n_train = len(y)
    print(
        f"  train rows={n_train}, counter labels={int(has_counter.sum())} "
        f"({100 * has_counter.mean():.1f}%)"
    )

    # outer CV split using canonical UMAP+Morgan+Jaccard
    splits = umap_split_indices(
        smiles, n_splits=args.outer_folds, n_clusters=50, seed=args.seed
    )

    # ----- Decomposition path -----
    print("\n===== Decomp path: y_hat = c_hat + s_hat =====")
    oof_decomp = np.full(n_train, np.nan, dtype=np.float64)
    fold_metrics_decomp: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        c_tr_label = c_label[tr_idx]
        has_c_tr = has_counter[tr_idx]

        # M_c training data: counter-labeled subset within fold_train
        X_c = X_tr[has_c_tr]
        c_c = c_tr_label[has_c_tr]

        # cross-fit c_hat for counter-labeled samples
        c_hat_oof = cross_val_predict_lgbm(
            X_c, c_c, seed=args.seed, n_splits=args.inner_folds
        )
        # full-fit M_c for non-counter and val
        m_c_full = fit_lgbm(X_c, c_c, seed=args.seed)

        # assemble c_hat for ALL fold_train rows
        c_hat_tr = np.empty(len(tr_idx), dtype=np.float64)
        c_hat_tr[has_c_tr] = c_hat_oof
        if (~has_c_tr).any():
            c_hat_tr[~has_c_tr] = m_c_full.predict(X_tr[~has_c_tr])
        c_hat_va = m_c_full.predict(X_va)

        # M_s on residual
        s_tr = y_tr - c_hat_tr
        m_s = fit_lgbm(X_tr, s_tr, seed=args.seed)
        s_hat_va = m_s.predict(X_va)

        y_hat_va = c_hat_va + s_hat_va
        oof_decomp[va_idx] = y_hat_va

        m = compute_metrics(y_va, y_hat_va)
        fold_metrics_decomp.append(m)
        print(
            f"  [decomp Fold {fold}] MAE={m['MAE']:.4f} RAE={m['RAE']:.4f} "
            f"R2={m['R2']:.4f} Sp={m['Spearman_R']:.4f}  "
            f"(M_c train n={len(c_c)}; M_s train n={len(tr_idx)})"
        )

    # ----- Direct baseline path -----
    print("\n===== Direct baseline: M_total directly on y =====")
    oof_direct = np.full(n_train, np.nan, dtype=np.float64)
    fold_metrics_direct: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_total = fit_lgbm(X[tr_idx], y[tr_idx], seed=args.seed)
        pred = m_total.predict(X[va_idx])
        oof_direct[va_idx] = pred
        m = compute_metrics(y[va_idx], pred)
        fold_metrics_direct.append(m)
        print(
            f"  [direct Fold {fold}] MAE={m['MAE']:.4f} RAE={m['RAE']:.4f} "
            f"R2={m['R2']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    # ----- Summary -----
    overall_decomp = compute_metrics(y, oof_decomp)
    overall_direct = compute_metrics(y, oof_direct)
    delta = overall_decomp["MAE"] - overall_direct["MAE"]

    print("\n===== Summary =====")
    print(
        f"  direct M_total   OOF MAE={overall_direct['MAE']:.4f}  "
        f"RAE={overall_direct['RAE']:.4f}  Sp={overall_direct['Spearman_R']:.4f}"
    )
    print(
        f"  decomp c+s       OOF MAE={overall_decomp['MAE']:.4f}  "
        f"RAE={overall_decomp['RAE']:.4f}  Sp={overall_decomp['Spearman_R']:.4f}"
    )
    gate1 = "PASS" if delta <= -0.005 else "fail"
    print(f"  Δ vs direct      {delta:+.4f}  [Phase 1A gate (<=-0.005): {gate1}]")
    print(f"  vs caruana_bag20 (0.4150) Δ = {overall_decomp['MAE'] - 0.4150:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature",
        choices=["rdkit_desc_full", "mordred"],
        default="rdkit_desc_full",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
