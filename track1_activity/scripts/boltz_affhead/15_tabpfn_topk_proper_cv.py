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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

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


def log2fc_pred_indices(feature_name: str, n_features: int) -> np.ndarray:
    if "log2fc_pred" not in feature_name:
        raise ValueError(
            "--exclude-log2fc-preds/--residual-log2fc-base require a feature set "
            "with predicted log2fc columns."
        )
    if n_features < 2:
        raise ValueError(f"Expected at least 2 features, got {n_features}")
    return np.array([n_features - 2, n_features - 1], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument(
        "--feature",
        type=str,
        default="cheme_2d_full_boltz_log2fc_pred",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tabpfn-version",
        choices=["v3", "v2_6", "v2_5", "v2"],
        default="v3",
        help="TabPFN checkpoint version. Defaults to TabPFN-3 OSS.",
    )
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--softmax-temperature", type=float, default=0.9)
    ap.add_argument("--average-before-softmax", action="store_true")
    ap.add_argument(
        "--exclude-log2fc-preds",
        action="store_true",
        help="Drop predicted log2fc columns before LGBM top-K selection and TabPFN.",
    )
    ap.add_argument(
        "--residual-log2fc-base",
        action="store_true",
        help=(
            "Fit a fold-local ridge base on predicted log2fc columns and train "
            "TabPFN on non-log2fc residuals."
        ),
    )
    args = ap.parse_args()
    if args.exclude_log2fc_preds and args.residual_log2fc_base:
        raise ValueError(
            "Use either --exclude-log2fc-preds or --residual-log2fc-base, not both."
        )

    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path

    import run_train

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

    X_tr_full, X_te_full = run_train.load_features(args.feature, tr_df, te_df)
    y = tr_df["pec50"].to_numpy(dtype=np.float32)

    col_mean = np.nanmean(X_tr_full, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_tr_full = np.where(np.isfinite(X_tr_full), X_tr_full, col_mean).astype(np.float32)
    X_te_full = np.where(np.isfinite(X_te_full), X_te_full, col_mean).astype(np.float32)

    log2fc_idx = np.array([], dtype=np.int64)
    if args.exclude_log2fc_preds or args.residual_log2fc_base:
        log2fc_idx = log2fc_pred_indices(args.feature, X_tr_full.shape[1])
        keep_mask = np.ones(X_tr_full.shape[1], dtype=bool)
        keep_mask[log2fc_idx] = False
        X_log2fc_tr = X_tr_full[:, log2fc_idx]
        X_log2fc_te = X_te_full[:, log2fc_idx]
        X_tr_model = X_tr_full[:, keep_mask]
        X_te_model = X_te_full[:, keep_mask]
    else:
        X_log2fc_tr = None
        X_log2fc_te = None
        X_tr_model = X_tr_full
        X_te_model = X_te_full

    print(
        f"Feature: {args.feature}  d={X_tr_full.shape[1]}  model_d={X_tr_model.shape[1]}  "
        f"train={X_tr_full.shape[0]}  test={X_te_full.shape[0]}"
    )
    print(f"K={args.K} (proper CV -- LGBM rank computed per-fold on train only)")
    if args.exclude_log2fc_preds:
        print(f"Dropped predicted log2fc columns: {log2fc_idx.tolist()}")
    if args.residual_log2fc_base:
        print(
            f"Residual mode: ridge base on predicted log2fc columns {log2fc_idx.tolist()}"
        )
    print(f"TabPFN version: {args.tabpfn_version}")
    print(
        "TabPFN params: "
        f"n_estimators={args.n_estimators}, "
        f"softmax_temperature={args.softmax_temperature}, "
        f"average_before_softmax={args.average_before_softmax}"
    )

    splits = umap_split_indices(
        tr_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )

    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        if args.residual_log2fc_base:
            assert X_log2fc_tr is not None
            assert X_log2fc_te is not None
            base = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-4, 4, 17)),
            )
            base.fit(X_log2fc_tr[tr_idx], y[tr_idx])
            base_tr = base.predict(X_log2fc_tr[tr_idx]).astype(np.float32)
            base_va = base.predict(X_log2fc_tr[va_idx]).astype(np.float32)
            base_te = base.predict(X_log2fc_te).astype(np.float32)
            fit_y = (y[tr_idx] - base_tr).astype(np.float32)
        else:
            base_va = None
            base_te = None
            fit_y = y[tr_idx]

        # Per-fold LGBM importance computed on tr_idx only (no leak)
        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=args.seed,
            verbose=-1,
        )
        lgbm.fit(X_tr_model[tr_idx], fit_y)
        gain = lgbm.booster_.feature_importance(importance_type="gain")
        sel = np.argsort(-gain)[: args.K]

        # TabPFN on selected features
        reg = TabPFNRegressor(
            device="cuda",
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
            average_before_softmax=args.average_before_softmax,
            random_state=args.seed,
            model_path=model_path,
            ignore_pretraining_limits=(args.K > 500),
        )
        reg.fit(X_tr_model[tr_idx][:, sel], fit_y)
        tabpfn_va = reg.predict(X_tr_model[va_idx][:, sel])
        tabpfn_te = reg.predict(X_te_model[:, sel])
        if args.residual_log2fc_base:
            assert base_va is not None
            assert base_te is not None
            oof[va_idx] = base_va + tabpfn_va
            test_preds_per_fold.append(base_te + tabpfn_te)
        else:
            oof[va_idx] = tabpfn_va
            test_preds_per_fold.append(tabpfn_te)

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
    if args.exclude_log2fc_preds:
        exp_name += "_nolog2fc"
    if args.residual_log2fc_base:
        exp_name += "_residlog2fc"
    if args.tabpfn_version != "v2_6":
        exp_name += f"_{args.tabpfn_version}"
    if args.n_estimators != 8:
        exp_name += f"_n{args.n_estimators}"
    if args.softmax_temperature != 0.9:
        exp_name += f"_temp{str(args.softmax_temperature).replace('.', 'p')}"
    if args.average_before_softmax:
        exp_name += "_avgpre"
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
            "n_estimators": args.n_estimators,
            "softmax_temperature": args.softmax_temperature,
            "average_before_softmax": args.average_before_softmax,
            "exclude_log2fc_preds": args.exclude_log2fc_preds,
            "residual_log2fc_base": args.residual_log2fc_base,
            "log2fc_pred_indices": log2fc_idx.tolist(),
            "residual_base_model": "StandardScaler+RidgeCV"
            if args.residual_log2fc_base
            else None,
            "ignore_pretraining_limits": args.K > 500,
            "tabpfn_version": args.tabpfn_version,
            "lgbm_n_estimators": 500,
            "lgbm_learning_rate": 0.05,
            "lgbm_num_leaves": 63,
            "seed": args.seed,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=(
            f"Top-{args.K} feature selection via per-fold LGBM gain, proper OOF. "
            f"exclude_log2fc_preds={args.exclude_log2fc_preds}; "
            f"residual_log2fc_base={args.residual_log2fc_base}. "
            f"TabPFN version={args.tabpfn_version}. "
            f"OOF MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f} "
            f"Sp={overall['Spearman_R']:.4f}"
        ),
    )
    save_oof_predictions(exp_id, oof)
    print(f"Recorded experiment id={exp_id}: {exp_name}")


if __name__ == "__main__":
    main()
