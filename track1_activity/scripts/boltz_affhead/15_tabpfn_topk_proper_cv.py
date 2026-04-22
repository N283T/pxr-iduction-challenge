"""Proper OOF evaluation of top-K LGBM-selected TabPFN.

Fixes the leak in 13_tabpfn_top_k_importance.py by moving the LGBM
feature-importance fit INSIDE the outer CV loop (each fold ranks
features using only its own training rows). This gives a leak-free OOF
MAE directly comparable to other pool members.

Default K=500 per the 13_ bakeoff finding (OOF 0.4169 vs production
0.4212, best K in the 2103 -> 500 sweep). Registers a proper experiment
with OOF + submission CSV so it can enter caruana immediately.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/15_tabpfn_topk_proper_cv.py --K 500
    pixi run python track1_activity/scripts/boltz_affhead/15_tabpfn_topk_proper_cv.py --K 800
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
from sklearn.metrics import mean_absolute_error, r2_score

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument(
        "--feature",
        type=str,
        default="cheme_2d_full_boltz_log2fc_pred",
    )
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
    print(f"K={args.K} (proper CV -- LGBM rank computed per-fold on train only)")

    splits = umap_split_indices(
        tr_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )

    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Per-fold LGBM importance computed on tr_idx only (no leak)
        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=args.seed,
            verbose=-1,
        )
        lgbm.fit(X_tr_full[tr_idx], y[tr_idx])
        gain = lgbm.booster_.feature_importance(importance_type="gain")
        sel = np.argsort(-gain)[: args.K]

        # TabPFN on selected features
        reg = TabPFNRegressor(
            device="cuda",
            n_estimators=8,
            random_state=args.seed,
            ignore_pretraining_limits=(args.K > 500),
        )
        reg.fit(X_tr_full[tr_idx][:, sel], y[tr_idx])
        oof[va_idx] = reg.predict(X_tr_full[va_idx][:, sel])
        test_preds_per_fold.append(reg.predict(X_te_full[:, sel]))

        m = compute_metrics(y[va_idx], oof[va_idx])
        fold_metrics.append(m)
        print(
            f"  [Fold {fold}] n_selected={args.K}  zero_gain_in_sel={(gain[sel] == 0).sum()}  "
            f"MAE={m['MAE']:.4f} RAE={m['RAE']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    overall = compute_metrics(y, oof)
    print("\nOverall OOF (proper CV):")
    for k_, v in overall.items():
        print(f"  {k_}={v:.4f}")

    # Reference
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        ref_name = f"tabpfn_{args.feature}_umap_default"
        cur.execute(
            "SELECT mae_mean, rae_mean, spearman_mean FROM experiment_summary WHERE name=%s",
            (ref_name,),
        )
        row = cur.fetchone()
        if row:
            ref_mae, ref_rae, ref_sp = float(row[0]), float(row[1]), float(row[2])
            print(
                f"\nReference {ref_name}: MAE={ref_mae:.4f} RAE={ref_rae:.4f} Sp={ref_sp:.4f}"
            )
            print(
                f"Δ (topK proper CV - full TabPFN): MAE={overall['MAE'] - ref_mae:+.4f}"
            )

    # Save submission + experiment
    test_preds_mean = np.mean(np.stack(test_preds_per_fold), axis=0)
    exp_name = f"tabpfn_{args.feature}_top{args.K}_umap"
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
            f"TabPFN on top-{args.K} LGBM-gain-ranked features of {args.feature}, "
            f"per-fold selection (no leak), UMAP 5-fold CV."
        ),
        model_type="tabpfn",
        feature_set=args.feature,
        hyperparameters={
            "K": args.K,
            "n_estimators": 8,
            "ignore_pretraining_limits": args.K > 500,
            "lgbm_n_estimators": 500,
            "lgbm_learning_rate": 0.05,
            "lgbm_num_leaves": 63,
            "seed": args.seed,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=(
            f"Top-{args.K} feature selection via per-fold LGBM gain, proper OOF. "
            f"OOF MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f} "
            f"Sp={overall['Spearman_R']:.4f}"
        ),
    )
    save_oof_predictions(exp_id, oof)
    print(f"Recorded experiment id={exp_id}: {exp_name}")


if __name__ == "__main__":
    main()
