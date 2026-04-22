"""Generalized top-K / PCA / PLS feature reduction → TabPFN bakeoff.

Extends 15_tabpfn_topk_proper_cv.py with a --reduction flag so we can
directly compare dim-reduction strategies on the same feature base:

  topk  (default): LGBM gain top-K (current best)
  pca            : unsupervised linear projection to K components
  pls            : supervised linear projection (uses y), K components

All done per-fold (fit on train rows only) so OOF is leak-free.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/17_tabpfn_dimreduce_proper_cv.py \\
        --feature cheme_2d_full_boltz_log2fc_pred --K 500 --reduction pca
    pixi run python track1_activity/scripts/boltz_affhead/17_tabpfn_dimreduce_proper_cv.py \\
        --feature kermt_pretrain_embed --K 500 --reduction topk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from evaluate import record_experiment, save_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def rae(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RAE": rae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman_R": float(spearmanr(y_pred, y_true).statistic),
        "Kendall_Tau": float(kendalltau(y_pred, y_true).statistic),
    }


def reduce_fold(
    reduction: str,
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    K: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit reduction on X_tr (+y_tr for supervised), transform all 3 splits."""
    if reduction == "topk":
        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=seed,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr)
        gain = lgbm.booster_.feature_importance(importance_type="gain")
        sel = np.argsort(-gain)[:K]
        return X_tr[:, sel], X_va[:, sel], X_te[:, sel]
    if reduction == "pca":
        # PCA is scale-sensitive; standardize features first
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)
        pca = PCA(n_components=K, random_state=seed)
        X_tr_r = pca.fit_transform(X_tr_s).astype(np.float32)
        X_va_r = pca.transform(X_va_s).astype(np.float32)
        X_te_r = pca.transform(X_te_s).astype(np.float32)
        return X_tr_r, X_va_r, X_te_r
    if reduction == "pls":
        # PLS uses y -- supervised
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)
        pls = PLSRegression(n_components=min(K, X_tr.shape[1], X_tr.shape[0] - 1))
        pls.fit(X_tr_s, y_tr)
        X_tr_r = pls.transform(X_tr_s).astype(np.float32)
        X_va_r = pls.transform(X_va_s).astype(np.float32)
        X_te_r = pls.transform(X_te_s).astype(np.float32)
        return X_tr_r, X_va_r, X_te_r
    raise ValueError(f"Unknown reduction: {reduction}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--feature", type=str, default="cheme_2d_full_boltz_log2fc_pred")
    ap.add_argument("--reduction", choices=["topk", "pca", "pls"], default="topk")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from tabpfn import TabPFNRegressor

    import run_train

    with psycopg2.connect(**DB_PARAMS) as conn:
        tr_df = pd.read_sql(
            """SELECT t.id AS compound_id, t.pec50, c.std_smiles AS smiles, c.molecule_name
                 FROM train_activity t JOIN compounds c ON c.id = t.id
                 ORDER BY t.id""",
            conn,
        )
        te_df = pd.read_sql(
            """SELECT t.id AS compound_id, c.std_smiles AS smiles, c.molecule_name
                 FROM test_activity t JOIN compounds c ON c.id = t.id
                 ORDER BY t.id""",
            conn,
        )

    X_tr_full, X_te_full = run_train.load_features(args.feature, tr_df, te_df)
    y = tr_df["pec50"].to_numpy(dtype=np.float32)

    col_mean = np.nanmean(X_tr_full, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_tr_full = np.where(np.isfinite(X_tr_full), X_tr_full, col_mean).astype(np.float32)
    X_te_full = np.where(np.isfinite(X_te_full), X_te_full, col_mean).astype(np.float32)

    print(
        f"Feature: {args.feature}  d={X_tr_full.shape[1]}  "
        f"train={X_tr_full.shape[0]}  test={X_te_full.shape[0]}"
    )
    print(f"Reduction: {args.reduction}  K={args.K} (per-fold, proper CV)")

    splits = umap_split_indices(
        tr_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )

    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        Xtr_r, Xva_r, Xte_r = reduce_fold(
            args.reduction,
            X_tr_full[tr_idx],
            X_tr_full[va_idx],
            X_te_full,
            y[tr_idx],
            args.K,
            args.seed,
        )

        reg = TabPFNRegressor(
            device="cuda",
            n_estimators=8,
            random_state=args.seed,
            ignore_pretraining_limits=(Xtr_r.shape[1] > 500),
        )
        reg.fit(Xtr_r, y[tr_idx])
        oof[va_idx] = reg.predict(Xva_r)
        test_preds_per_fold.append(reg.predict(Xte_r))

        m = compute_metrics(y[va_idx], oof[va_idx])
        fold_metrics.append(m)
        print(
            f"  [Fold {fold}] d_out={Xtr_r.shape[1]}  "
            f"MAE={m['MAE']:.4f} RAE={m['RAE']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    overall = compute_metrics(y, oof)
    print(f"\nOverall OOF ({args.reduction}, K={args.K}):")
    for k_, v in overall.items():
        print(f"  {k_}={v:.4f}")

    # Compare against references
    refs = [
        f"tabpfn_{args.feature}_umap_default",
        f"tabpfn_{args.feature}_top{args.K}_umap",
    ]
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        print("\nReferences:")
        for ref in refs:
            cur.execute(
                "SELECT mae_mean, rae_mean, spearman_mean FROM experiment_summary WHERE name=%s",
                (ref,),
            )
            row = cur.fetchone()
            if row:
                ref_mae = float(row[0])
                print(f"  {ref}: MAE={ref_mae:.4f}  Δ={overall['MAE'] - ref_mae:+.4f}")

    # Save experiment + submission
    test_preds_mean = np.mean(np.stack(test_preds_per_fold), axis=0)
    exp_name = f"tabpfn_{args.feature}_{args.reduction}{args.K}_umap"
    sub = pd.DataFrame(
        {
            "SMILES": te_df["smiles"],
            "Molecule Name": te_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")

    exp_id = record_experiment(
        name=exp_name,
        description=(
            f"TabPFN on {args.reduction}-reduced (K={args.K}) {args.feature}, "
            f"per-fold reduction, UMAP 5-fold CV."
        ),
        model_type="tabpfn",
        feature_set=args.feature,
        hyperparameters={
            "K": args.K,
            "reduction": args.reduction,
            "n_estimators": 8,
            "seed": args.seed,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=(
            f"Reduction={args.reduction} K={args.K}  "
            f"OOF MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f} "
            f"Sp={overall['Spearman_R']:.4f}"
        ),
    )
    save_oof_predictions(exp_id, oof)
    print(f"Recorded experiment id={exp_id}: {exp_name}")


if __name__ == "__main__":
    main()
