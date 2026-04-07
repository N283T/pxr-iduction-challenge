#!/usr/bin/env -S pixi run python
"""Residual learning: physicochemical base + Mordred/FP residual.

Stage 1: Shallow GBDT on key physicochemical properties (logP, MW, TPSA, etc.)
Stage 2: Full Mordred model learns residual (y - stage1_pred)
Final: stage1_pred + stage2_pred

Motivation: PXR models are dominated by logP. Stage 1 captures the
physicochemical "shortcut", Stage 2 learns structural details that
generalize better to novel chemotypes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from data import (
    DESCRIPTOR_COLS,
    load_train_descriptors,
    load_train_mordred,
    load_test_descriptors,
    load_test_smiles,
    load_train_smiles_target,
    get_conn,
    load_mordred,
)
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices, scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Key physicochemical properties for Stage 1
PHYSPROP_COLS = [
    "logp",
    "amw",
    "tpsa",
    "hba",
    "hbd",
    "num_rotatable_bonds",
    "num_aromatic_rings",
    "fractioncsp3",
    "labuteasa",
    "num_heavy_atoms",
]


SPLIT_TABLES = {"train": "train_activity", "test": "test_activity"}


def load_compound_ids(split: str) -> list[int]:
    table = SPLIT_TABLES[split]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def tune_and_train(X_tr, y_tr, X_va, y_va, n_trials=20, shallow=False):
    """Tune LightGBM and return (val_preds, train_preds, best_iter, params)."""

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": 42,
            "num_leaves": trial.suggest_int("num_leaves", 7, 63 if not shallow else 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        if shallow:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 6)

        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)
        model = lgb.train(
            params,
            dt,
            num_boost_round=1000 if not shallow else 500,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return np.mean(np.abs(y_va - model.predict(X_va)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 42,
        **study.best_params,
    }

    dt = lgb.Dataset(X_tr, label=y_tr)
    dv = lgb.Dataset(X_va, label=y_va, reference=dt)
    model = lgb.train(
        best_params,
        dt,
        num_boost_round=1000 if not shallow else 500,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    val_preds = model.predict(X_va)
    train_preds = model.predict(X_tr)
    return val_preds, train_preds, model.best_iteration, best_params, model


def main():
    parser = argparse.ArgumentParser(description="Residual learning")
    parser.add_argument("--split", choices=["scaffold", "umap"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_desc = load_train_descriptors()
    test_desc = load_test_descriptors()
    y_train = train_df["pec50"].values

    # Stage 1 features: physicochemical properties
    X_phys_train = train_desc[PHYSPROP_COLS].values.astype(np.float32)
    X_phys_test = test_desc[PHYSPROP_COLS].values.astype(np.float32)

    # Stage 2 features: Mordred
    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")
    mordred_train, _ = load_train_mordred()
    mordred_test = load_mordred(test_ids)
    common_cols = mordred_train.columns.intersection(mordred_test.columns)
    X_mord_train = mordred_train.loc[train_ids, common_cols].values.astype(np.float32)
    X_mord_test = mordred_test.loc[test_ids, common_cols].values.astype(np.float32)
    X_mord_train = np.nan_to_num(X_mord_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_mord_test = np.nan_to_num(X_mord_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Stage 1 features: {X_phys_train.shape[1]} (physicochemical)")
    print(f"Stage 2 features: {X_mord_train.shape[1]} (Mordred)")

    # CV splits
    smiles_list = train_df["smiles"].tolist()
    if args.split == "umap":
        outer_splits = umap_split_indices(
            smiles_list,
            n_splits=5,
            n_clusters=args.n_clusters,
            seed=42,
        )
    else:
        outer_splits = scaffold_split_indices(smiles_list, n_splits=5, seed=42)

    exp_name = f"residual_physprop+mordred_{args.split}"
    print(f"Experiment: {exp_name}")
    print(f"Split: {args.split}, Trials: {args.trials}")

    oof_preds_s1 = np.zeros(len(y_train))
    oof_preds_final = np.zeros(len(y_train))
    test_preds_s1_all = []
    test_preds_s2_all = []
    fold_metrics_s1 = []
    fold_metrics_final = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        print(f"\n--- Fold {fold} ---")

        # Stage 1: physicochemical base model (shallow)
        print("  Stage 1: physicochemical properties...")
        s1_va, s1_tr, s1_iter, s1_params, s1_model = tune_and_train(
            X_phys_train[tr_idx],
            y_train[tr_idx],
            X_phys_train[va_idx],
            y_train[va_idx],
            n_trials=args.trials,
            shallow=True,
        )
        oof_preds_s1[va_idx] = s1_va
        m_s1 = compute_metrics(y_train[va_idx], s1_va)
        fold_metrics_s1.append(m_s1)
        print_metrics(m_s1, label="Stage 1")

        # Stage 1 test prediction
        s1_test = s1_model.predict(X_phys_test)
        test_preds_s1_all.append(s1_test)

        # Compute residuals
        residual_tr = y_train[tr_idx] - s1_tr
        residual_va = y_train[va_idx] - s1_va

        # Stage 2: Mordred on residuals
        print("  Stage 2: Mordred on residuals...")
        s2_va, s2_tr, s2_iter, s2_params, s2_model = tune_and_train(
            X_mord_train[tr_idx],
            residual_tr,
            X_mord_train[va_idx],
            residual_va,
            n_trials=args.trials,
            shallow=False,
        )

        # Final prediction
        final_va = s1_va + s2_va
        oof_preds_final[va_idx] = final_va
        m_final = compute_metrics(y_train[va_idx], final_va)
        fold_metrics_final.append(m_final)
        print_metrics(m_final, label="Final (S1+S2)")

        # Stage 2 test prediction
        s2_test = s2_model.predict(X_mord_test)
        test_preds_s2_all.append(s2_test)

    # Overall OOF
    print(f"\n{'=' * 60}")
    oof_s1 = compute_metrics(y_train, oof_preds_s1)
    print("  Stage 1 OOF:")
    print_metrics(oof_s1)

    oof_final = compute_metrics(y_train, oof_preds_final)
    print("  Final OOF (S1+S2):")
    print_metrics(oof_final)
    print_fold_summary(fold_metrics_final)

    # Test predictions: average across folds
    test_s1 = np.mean(test_preds_s1_all, axis=0)
    test_s2 = np.mean(test_preds_s2_all, axis=0)
    test_final = test_s1 + test_s2

    # Save submission
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_final,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    # Record
    exp_id = record_experiment(
        name=exp_name,
        description="Residual learning: physicochemical Stage 1 + Mordred Stage 2",
        model_type="residual_lgbm",
        feature_set="physprop+mordred_residual",
        hyperparameters={"stage1_features": PHYSPROP_COLS, "stage2": "mordred_2d"},
        fold_metrics=fold_metrics_final,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=f"OOF RAE={oof_final['RAE']:.4f}, S1 RAE={oof_s1['RAE']:.4f}, {args.split}_split",
    )
    save_oof_predictions(exp_id, oof_preds_final)

    print(f"\n  Stage 1 alone: RAE={oof_s1['RAE']:.4f}")
    print(f"  Final (S1+S2): RAE={oof_final['RAE']:.4f}")
    print(f"  Improvement:   {oof_s1['RAE'] - oof_final['RAE']:.4f}")


if __name__ == "__main__":
    main()
