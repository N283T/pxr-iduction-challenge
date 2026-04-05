#!/usr/bin/env -S pixi run python
"""Train CatBoost, XGBoost, and TabPFN models for ensemble diversity.

Each model uses mordred features (best single feature set),
with Optuna hyperparameter tuning for CatBoost/XGBoost.
Results and OOF predictions are recorded to DB for ensemble.

Addresses issue #19.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from data import (
    DESCRIPTOR_COLS,
    load_test_descriptors,
    load_test_smiles,
    load_train_descriptors,
    load_train_mordred,
    load_train_smiles_target,
    get_conn,
)
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from features import FP_REGISTRY, smiles_to_mols
from splits import scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_OPTUNA_TRIALS = 20
INNER_CV_FOLDS = 3


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_mordred_aligned():
    """Load mordred features aligned with train_activity ordering."""
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    mordred_train, _ = load_train_mordred()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    train_cids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT compound_id FROM test_activity ORDER BY id")
    test_cids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    from data import load_mordred

    mordred_test = load_mordred(test_cids)

    # Align columns
    common_cols = mordred_train.columns.intersection(mordred_test.columns)
    X_train = mordred_train.loc[train_cids, common_cols].values.astype(np.float32)
    X_test = mordred_test.loc[test_cids, common_cols].values.astype(np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = train_df["pec50"].values
    return X_train, y_train, X_test, train_df, test_df


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def run_catboost(X_train, y_train, X_test, test_df, outer_splits):
    """CatBoost with Optuna tuning + scaffold CV."""
    print(f"\n{'=' * 60}")
    print("  CatBoost (Mordred features)")
    print(f"{'=' * 60}")

    def create_objective(X_fold, y_fold):
        def objective(trial):
            params = {
                "iterations": 2000,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-3, 10.0, log=True
                ),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "loss_function": "MAE",
                "eval_metric": "MAE",
                "verbose": 0,
                "random_seed": 42,
                "allow_writing_files": False,
            }
            kf = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=42)
            scores = []
            for ti, vi in kf.split(X_fold):
                pool_train = cb.Pool(X_fold[ti], label=y_fold[ti])
                pool_val = cb.Pool(X_fold[vi], label=y_fold[vi])
                model = cb.CatBoostRegressor(**params)
                model.fit(pool_train, eval_set=pool_val, early_stopping_rounds=50)
                preds = model.predict(X_fold[vi])
                scores.append(np.mean(np.abs(y_fold[vi] - preds)))
            return np.mean(scores)

        return objective

    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    num_boost_rounds = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        print(f"  [Fold {fold}] Tuning CatBoost...")
        study = optuna.create_study(direction="minimize")
        study.optimize(create_objective(X_tr, y_tr), n_trials=N_OPTUNA_TRIALS)

        best_params = {
            "iterations": 2000,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "verbose": 0,
            "random_seed": 42,
            "allow_writing_files": False,
            **study.best_params,
        }

        pool_train = cb.Pool(X_tr, label=y_tr)
        pool_val = cb.Pool(X_va, label=y_va)
        model = cb.CatBoostRegressor(**best_params)
        model.fit(pool_train, eval_set=pool_val, early_stopping_rounds=50)

        vp = model.predict(X_va)
        oof_preds[va_idx] = vp
        metrics = compute_metrics(y_va, vp)
        fold_metrics.append(metrics)
        num_boost_rounds.append(model.best_iteration_)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model for test predictions
    print("  Training final CatBoost model...")
    final_study = optuna.create_study(direction="minimize")
    final_study.optimize(create_objective(X_train, y_train), n_trials=N_OPTUNA_TRIALS)
    final_params = {
        "iterations": int(np.mean(num_boost_rounds)),
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
        **final_study.best_params,
    }
    final_model = cb.CatBoostRegressor(**final_params)
    final_model.fit(cb.Pool(X_train, label=y_train))
    test_preds = final_model.predict(X_test)

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub.to_csv(SUBMISSION_DIR.joinpath("catboost_mordred.csv"), index=False)

    exp_id = record_experiment(
        name="catboost_mordred",
        description="CatBoost with Optuna tuning on Mordred features",
        model_type="catboost",
        feature_set="mordred_2d",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path="track1_activity/submissions/catboost_mordred.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, scaffold_split, nested_cv",
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def run_xgboost(X_train, y_train, X_test, test_df, outer_splits):
    """XGBoost with Optuna tuning + scaffold CV."""
    print(f"\n{'=' * 60}")
    print("  XGBoost (Mordred features)")
    print(f"{'=' * 60}")

    def create_objective(X_fold, y_fold):
        def objective(trial):
            params = {
                "objective": "reg:absoluteerror",
                "eval_metric": "mae",
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "seed": 42,
                "verbosity": 0,
            }
            kf = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=42)
            scores = []
            for ti, vi in kf.split(X_fold):
                dtrain = xgb.DMatrix(X_fold[ti], label=y_fold[ti])
                dval = xgb.DMatrix(X_fold[vi], label=y_fold[vi])
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=2000,
                    evals=[(dval, "val")],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
                preds = model.predict(dval)
                scores.append(np.mean(np.abs(y_fold[vi] - preds)))
            return np.mean(scores)

        return objective

    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    num_boost_rounds = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        print(f"  [Fold {fold}] Tuning XGBoost...")
        study = optuna.create_study(direction="minimize")
        study.optimize(create_objective(X_tr, y_tr), n_trials=N_OPTUNA_TRIALS)

        best_params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "seed": 42,
            "verbosity": 0,
            **study.best_params,
        }

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        vp = model.predict(dval)
        oof_preds[va_idx] = vp
        metrics = compute_metrics(y_va, vp)
        fold_metrics.append(metrics)
        num_boost_rounds.append(model.best_iteration)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model
    print("  Training final XGBoost model...")
    final_study = optuna.create_study(direction="minimize")
    final_study.optimize(create_objective(X_train, y_train), n_trials=N_OPTUNA_TRIALS)
    final_params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
        **final_study.best_params,
    }
    avg_rounds = int(np.mean(num_boost_rounds))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    final_model = xgb.train(final_params, dtrain, num_boost_round=avg_rounds)
    test_preds = final_model.predict(xgb.DMatrix(X_test))

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub.to_csv(SUBMISSION_DIR.joinpath("xgboost_mordred.csv"), index=False)

    exp_id = record_experiment(
        name="xgboost_mordred",
        description="XGBoost with Optuna tuning on Mordred features",
        model_type="xgboost",
        feature_set="mordred_2d",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path="track1_activity/submissions/xgboost_mordred.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, scaffold_split, nested_cv",
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


# ---------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------


def run_tabpfn(X_train, y_train, X_test, test_df, outer_splits):
    """TabPFN (prior-fitted network) with scaffold CV.

    TabPFN is pre-trained and requires no hyperparameter tuning.
    It has a limit on dataset size, so we may need to subsample features.
    """
    print(f"\n{'=' * 60}")
    print("  TabPFN (Mordred features)")
    print(f"{'=' * 60}")

    from tabpfn import TabPFNRegressor

    # TabPFN has limits on features — reduce dimensionality if needed
    from sklearn.decomposition import PCA

    n_components = min(100, X_train.shape[1])
    print(f"  Reducing {X_train.shape[1]} features to {n_components} via PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    oof_preds = np.zeros(len(y_train))
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        X_tr, X_va = X_train_pca[tr_idx], X_train_pca[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        print(f"  [Fold {fold}] Fitting TabPFN (n_train={len(tr_idx)})...")
        model = TabPFNRegressor(
            n_estimators=8,
            ignore_pretraining_limits=True,
            device="cuda",
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        vp = model.predict(X_va)
        oof_preds[va_idx] = vp
        metrics = compute_metrics(y_va, vp)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model for test
    print("  Training final TabPFN model...")
    final_model = TabPFNRegressor(
        n_estimators=8,
        ignore_pretraining_limits=True,
        device="cuda",
        random_state=42,
    )
    final_model.fit(X_train_pca, y_train)
    test_preds = final_model.predict(X_test_pca)

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub.to_csv(SUBMISSION_DIR.joinpath("tabpfn_mordred.csv"), index=False)

    exp_id = record_experiment(
        name="tabpfn_mordred",
        description="TabPFN on Mordred features (PCA-reduced)",
        model_type="tabpfn",
        feature_set="mordred_2d_pca100",
        hyperparameters={"n_estimators": 8, "n_pca_components": n_components},
        fold_metrics=fold_metrics,
        submission_path="track1_activity/submissions/tabpfn_mordred.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, scaffold_split, pca_{n_components}",
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading data...")
    X_train, y_train, X_test, train_df, test_df = load_mordred_aligned()
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    outer_splits = scaffold_split_indices(
        train_df["smiles"].tolist(), n_splits=5, seed=42
    )

    results = {}

    # CatBoost
    results["catboost"] = run_catboost(X_train, y_train, X_test, test_df, outer_splits)

    # XGBoost
    results["xgboost"] = run_xgboost(X_train, y_train, X_test, test_df, outer_splits)

    # TabPFN
    results["tabpfn"] = run_tabpfn(X_train, y_train, X_test, test_df, outer_splits)

    # Summary
    print(f"\n{'=' * 60}")
    print("  NEW MODELS SUMMARY")
    print(f"{'=' * 60}")
    for name, m in sorted(results.items(), key=lambda x: x[1]["RAE"]):
        print(
            f"  {name:<20} RAE={m['RAE']:.4f}  MAE={m['MAE']:.4f}  "
            f"R2={m['R2']:.4f}  Spearman={m['Spearman_R']:.4f}"
        )

    print(f"\n  For reference, best existing single model:")
    print(f"  single_mordred (LightGBM): RAE=0.5654")


if __name__ == "__main__":
    main()
