"""Bakeoff: cheme_2d_full_boltz_log2fc_pred (2103d) truncated to top-K
features by LGBM gain importance, then TabPFN 5-fold UMAP CV.

Motivation: user intuition that 2103d may contain low-importance 2D
descriptors that bloat the input and dilute TabPFN's attention. If true,
keeping only top-K (K=2000, 1500, 1000) should improve or match MAE.

Caveat: LGBM-importance-based selection uses y labels => mild leak. Kept
minimal (single fit on full train, K fixed) since the goal is just to
test the "TabPFN wins from quality dim" hypothesis, not to claim a
pure-OOF benchmark. If a K beats baseline, we can redo with CV-internal
selection as a follow-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402


def load_cheme_2df(
    train_ids: list[int], test_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct cheme_2d_full_boltz_log2fc_pred (2103d) for given ids.

    Uses run_train.load_features indirectly by importing and calling it.
    """
    # Call run_train's loader for cheme_2d_full_boltz_log2fc_pred
    sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
    import run_train  # noqa: E402

    # load_features needs (name, train_df, test_df) but they have specific cols
    with psycopg2.connect(**DB_PARAMS) as conn:
        tr_df = pd.read_sql(
            """SELECT t.id AS compound_id, t.pec50, c.std_smiles AS smiles, c.molecule_name
                 FROM train_activity t JOIN compounds c ON c.id = t.compound_id
                 ORDER BY t.id""",
            conn,
        )
        te_df = pd.read_sql(
            """SELECT t.id AS compound_id, c.std_smiles AS smiles, c.molecule_name
                 FROM test_activity t JOIN compounds c ON c.id = t.compound_id
                 ORDER BY t.id""",
            conn,
        )
    X_train, X_test = run_train.load_features(
        "cheme_2d_full_boltz_log2fc_pred", tr_df, te_df
    )
    return X_train, X_test, tr_df


def rae(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RAE": rae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_pred, y_true).statistic),
        "Kendall": float(kendalltau(y_pred, y_true).statistic),
    }


def tabpfn_5fold(X_train, X_test, y, train_smiles, tag: str) -> tuple[dict, np.ndarray]:
    from tabpfn import TabPFNRegressor

    splits = umap_split_indices(train_smiles, n_splits=5, n_clusters=50, seed=42)
    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        reg = TabPFNRegressor(
            device="cuda",
            n_estimators=8,
            random_state=42,
            ignore_pretraining_limits=True,
        )
        reg.fit(X_train[tr_idx], y[tr_idx])
        oof[va_idx] = reg.predict(X_train[va_idx])
        test_preds_per_fold.append(reg.predict(X_test))
        print(
            f"    [{tag} Fold {fold}] val MAE={mean_absolute_error(y[va_idx], oof[va_idx]):.4f}"
        )
    return compute_metrics(y, oof), np.mean(np.stack(test_preds_per_fold), axis=0)


def main() -> None:
    print("Loading cheme_2d_full_boltz_log2fc_pred (2103d) ...")
    with psycopg2.connect(**DB_PARAMS) as conn:
        tr_df = pd.read_sql("""SELECT t.id FROM train_activity t ORDER BY t.id""", conn)
        te_df = pd.read_sql("""SELECT t.id FROM test_activity t ORDER BY t.id""", conn)
    train_ids = tr_df["id"].tolist()
    test_ids = te_df["id"].tolist()
    X_tr, X_te, train_full = load_cheme_2df(train_ids, test_ids)
    y = train_full["pec50"].to_numpy(dtype=np.float32)
    train_smiles = train_full["smiles"].tolist()
    print(f"  X_train {X_tr.shape}, X_test {X_te.shape}")

    # Sanitize non-finite (TabPFN rejects NaN/Inf)
    col_mean_tr = np.nanmean(X_tr, axis=0)
    col_mean_tr = np.where(np.isfinite(col_mean_tr), col_mean_tr, 0.0)
    X_tr = np.where(np.isfinite(X_tr), X_tr, col_mean_tr)
    X_te = np.where(np.isfinite(X_te), X_te, col_mean_tr)

    print("\nFitting LGBM for feature importance ...")
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_tr, y)
    importance_gain = lgbm.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-importance_gain)

    print(f"  top 5 gain importance: {importance_gain[order[:5]].tolist()}")
    print(f"  zero-importance dims (gain): {(importance_gain == 0).sum()}")

    results: list[dict] = []
    ks_to_try = [2103, 2000, 1500, 1200, 1000, 800, 500]
    for K in ks_to_try:
        if K > X_tr.shape[1]:
            continue
        sel = order[:K]
        Xtr_k = X_tr[:, sel]
        Xte_k = X_te[:, sel]
        print(f"\n=== K={K} (top {K} by LGBM gain) ===")
        m, _ = tabpfn_5fold(Xtr_k, Xte_k, y, train_smiles, tag=f"K{K}")
        print(f"  K={K}: MAE={m['MAE']:.4f} RAE={m['RAE']:.4f} Sp={m['Spearman']:.4f}")
        results.append({"K": K, **m})

    print("\n" + "=" * 60)
    print(f"{'K':<6} {'MAE':<8} {'ΔMAE':<9} {'RAE':<8} {'Sp':<8}")
    print("-" * 60)
    base_mae = next(r["MAE"] for r in results if r["K"] == 2103)
    for r in results:
        delta = r["MAE"] - base_mae
        marker = " ⭐" if delta < -1e-4 else "  " if delta < 1e-4 else " ✗"
        print(
            f"{r['K']:<6} {r['MAE']:.4f}  {delta:+.4f}{marker} {r['RAE']:.4f}  {r['Spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
