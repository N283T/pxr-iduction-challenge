"""Nested CV with Optuna hyperparameter tuning.

Outer: scaffold split 5-fold
Inner: Optuna tunes LightGBM params within each outer train fold
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import psycopg2
from mordred import Calculator
from mordred import descriptors as mordred_descs
from sklearn.model_selection import KFold

from data import (
    DB_PARAMS,
    DESCRIPTOR_COLS,
    load_test_descriptors,
    load_test_smiles,
    load_train_descriptors,
    load_train_smiles_target,
)
from evaluate import compute_metrics, print_metrics, print_fold_summary, record_experiment
from features import FP_REGISTRY, smiles_to_mols
from splits import scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_OPTUNA_TRIALS = 50
INNER_CV_FOLDS = 3


def load_embeddings(table: str, compound_ids: list[int]) -> np.ndarray:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(compound_ids))
    cur.execute(
        f"SELECT compound_id, embedding FROM {table} WHERE compound_id IN ({placeholders})",
        compound_ids,
    )
    rows = {cid: emb for cid, emb in cur.fetchall()}
    cur.close()
    conn.close()
    return np.array([rows[cid] for cid in compound_ids])


def load_compound_ids(split: str) -> list[int]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    table = "train_activity" if split == "train" else "test_activity"
    cur.execute(f"SELECT compound_id FROM {table} ORDER BY compound_id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def create_objective(X_train_fold, y_train_fold):
    """Create Optuna objective for a given outer train fold."""

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": 42,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

        inner_kf = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=42)
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_kf.split(X_train_fold):
            X_itr = X_train_fold[inner_train_idx]
            y_itr = y_train_fold[inner_train_idx]
            X_ival = X_train_fold[inner_val_idx]
            y_ival = y_train_fold[inner_val_idx]

            dtrain = lgb.Dataset(X_itr, label=y_itr)
            dval = lgb.Dataset(X_ival, label=y_ival, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            pred = model.predict(X_ival)
            mae = np.mean(np.abs(y_ival - pred))
            inner_scores.append(mae)

        return np.mean(inner_scores)

    return objective


def run_nested_cv(name, X_train, y_train, X_test, test_df, feature_set, description, outer_splits):
    """Run nested CV: Optuna inner, scaffold outer."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({X_train.shape[1]} features)")
    print(f"  Outer: {len(outer_splits)} folds, Inner: {INNER_CV_FOLDS} folds x {N_OPTUNA_TRIALS} trials")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    fold_best_params = []
    num_boost_rounds = []

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Inner: Optuna tuning
        print(f"\n  [Fold {fold}] Tuning {N_OPTUNA_TRIALS} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(create_objective(X_tr, y_tr), n_trials=N_OPTUNA_TRIALS)

        best_params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": 42,
            **study.best_params,
        }
        fold_best_params.append(study.best_params)
        print(f"  [Fold {fold}] Best inner MAE: {study.best_value:.4f}")

        # Re-train on full outer train with best params
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            best_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        metrics = compute_metrics(y_val, val_pred)
        fold_metrics.append(metrics)
        num_boost_rounds.append(model.best_iteration)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model: tune on all train data, then train
    print("\n  Tuning final model on all data...")
    final_study = optuna.create_study(direction="minimize")
    final_study.optimize(create_objective(X_train, y_train), n_trials=N_OPTUNA_TRIALS)
    final_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 42,
        **final_study.best_params,
    }
    avg_rounds = int(np.mean(num_boost_rounds))
    final_model = lgb.train(
        final_params, lgb.Dataset(X_train, label=y_train), num_boost_round=avg_rounds
    )

    test_preds = final_model.predict(X_test)
    print(f"\n  Test preds: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    submission = pd.DataFrame({
        "SMILES": test_df["smiles"],
        "Molecule Name": test_df["molecule_name"],
        "pEC50": test_preds,
    })
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)

    # Merge all per-fold best params for notes
    all_hyperparams = {**final_params, "per_fold_params": fold_best_params}

    record_experiment(
        name=name,
        description=description,
        model_type="lightgbm_optuna",
        feature_set=feature_set,
        hyperparameters=all_hyperparams,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, nested_cv, scaffold_split, {N_OPTUNA_TRIALS} trials",
    )
    return oof_metrics


def main():
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_desc_df = load_train_descriptors()
    test_desc_df = load_test_descriptors()
    y_train = train_df["pec50"].values

    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")

    print("Computing features...")
    train_mols = smiles_to_mols(train_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])

    X_desc_tr = train_desc_df[DESCRIPTOR_COLS].values
    X_desc_te = test_desc_df[DESCRIPTOR_COLS].values
    X_morgan_tr = FP_REGISTRY["morgan_r2_2048"](train_mols)
    X_morgan_te = FP_REGISTRY["morgan_r2_2048"](test_mols)
    X_chemeleon_tr = load_embeddings("compound_chemeleon", train_ids)
    X_chemeleon_te = load_embeddings("compound_chemeleon", test_ids)

    # Mordred
    print("Computing Mordred descriptors...")
    calc = Calculator(mordred_descs, ignore_3D=True)
    train_mord_df = calc.pandas(train_mols, quiet=True)
    for col in train_mord_df.columns:
        train_mord_df[col] = pd.to_numeric(train_mord_df[col], errors="coerce")
    train_mord_df = train_mord_df.dropna(axis=1, how="all")
    train_mord_df = train_mord_df.loc[:, train_mord_df.nunique() > 1]
    train_mord_df = train_mord_df.replace([np.inf, -np.inf], np.nan)
    mordred_medians = train_mord_df.median()
    mordred_cols = list(train_mord_df.columns)
    train_mord_df = train_mord_df.fillna(mordred_medians)
    X_mordred_tr = train_mord_df.values.astype(np.float32)

    test_mord_df = calc.pandas(test_mols, quiet=True)
    for col in test_mord_df.columns:
        test_mord_df[col] = pd.to_numeric(test_mord_df[col], errors="coerce")
    test_mord_df = test_mord_df.reindex(columns=mordred_cols)
    test_mord_df = test_mord_df.replace([np.inf, -np.inf], np.nan)
    test_mord_df = test_mord_df.fillna(mordred_medians[mordred_cols])
    X_mordred_te = test_mord_df.values.astype(np.float32)

    # Scaffold splits
    print("Computing scaffold splits...")
    outer_splits = scaffold_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    # Feature configs
    configs = {
        "optuna_desc+morgan": {
            "X_train": np.hstack([X_desc_tr, X_morgan_tr]),
            "X_test": np.hstack([X_desc_te, X_morgan_te]),
            "feature_set": "rdkit_descriptors+morgan_r2_2048",
        },
        "optuna_chemeleon+desc+morgan": {
            "X_train": np.hstack([X_chemeleon_tr, X_desc_tr, X_morgan_tr]),
            "X_test": np.hstack([X_chemeleon_te, X_desc_te, X_morgan_te]),
            "feature_set": "chemeleon_300d+rdkit_descriptors+morgan_r2_2048",
        },
        "optuna_mordred+morgan": {
            "X_train": np.hstack([X_mordred_tr, X_morgan_tr]),
            "X_test": np.hstack([X_mordred_te, X_morgan_te]),
            "feature_set": "mordred_2d+morgan_r2_2048",
        },
    }

    results = {}
    for config_name, config in configs.items():
        m = run_nested_cv(
            config_name,
            config["X_train"],
            y_train,
            config["X_test"],
            test_df,
            config["feature_set"],
            f"LightGBM Optuna nested CV ({config_name})",
            outer_splits,
        )
        results[config_name] = m

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY (Optuna Nested CV, Scaffold Split)")
    print(f"{'=' * 70}")
    for name, m in sorted(results.items(), key=lambda x: x[1]["RAE"]):
        print(f"  {name:<40} RAE={m['RAE']:.4f}  R2={m['R2']:.4f}  Spearman={m['Spearman_R']:.4f}")


if __name__ == "__main__":
    main()
