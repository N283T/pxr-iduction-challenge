"""Run LightGBM experiments with ChemBERTa embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import (
    DESCRIPTOR_COLS,
    load_test_descriptors,
    load_test_smiles,
    load_train_chemberta,
    load_train_descriptors,
    load_train_smiles_target,
    load_test_chemberta,
)
from evaluate import compute_metrics, print_metrics, print_fold_summary, record_experiment
from features import FP_REGISTRY, smiles_to_mols

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}


def run_experiment(name, X_train, y_train, X_test, test_df, feature_set, description):
    """Run 5-fold CV, train final model, save submission, record to DB."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({X_train.shape[1]} features)")
    print(f"{'=' * 60}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    num_boost_rounds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
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

    avg_rounds = int(np.mean(num_boost_rounds))
    final_model = lgb.train(
        LGB_PARAMS, lgb.Dataset(X_train, label=y_train), num_boost_round=avg_rounds
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
    print(f"  Saved: {sub_path.name}")

    record_experiment(
        name=name,
        description=description,
        model_type="lightgbm",
        feature_set=feature_set,
        hyperparameters=LGB_PARAMS,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, avg_rounds={avg_rounds}",
    )
    return oof_metrics


def main():
    print("Loading data...")
    test_df = load_test_smiles()

    # ChemBERTa embeddings from DB
    train_emb, y_train = load_train_chemberta()
    test_emb = load_test_chemberta()
    X_train_emb = train_emb.values
    X_test_emb = test_emb.values
    print(f"  ChemBERTa: train={X_train_emb.shape}, test={X_test_emb.shape}")

    # --- ChemBERTa solo ---
    run_experiment(
        "lgbm_chemberta",
        X_train_emb, y_train.values, X_test_emb, test_df,
        "chemberta_384", "LightGBM with ChemBERTa-77M embeddings (384d)",
    )

    # --- ChemBERTa + RDKit descriptors ---
    train_desc_df = load_train_descriptors()
    test_desc_df = load_test_descriptors()
    X_train_desc = train_desc_df[DESCRIPTOR_COLS].values
    X_test_desc = test_desc_df[DESCRIPTOR_COLS].values

    X_train_combo = np.hstack([X_train_emb, X_train_desc])
    X_test_combo = np.hstack([X_test_emb, X_test_desc])

    run_experiment(
        "lgbm_chemberta+desc",
        X_train_combo, y_train.values, X_test_combo, test_df,
        "chemberta_384+rdkit_descriptors",
        "LightGBM with ChemBERTa embeddings + RDKit descriptors",
    )

    # --- ChemBERTa + Morgan FP ---
    train_df = load_train_smiles_target()
    train_mols = smiles_to_mols(train_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])

    X_train_morgan = FP_REGISTRY["morgan_r2_2048"](train_mols)
    X_test_morgan = FP_REGISTRY["morgan_r2_2048"](test_mols)

    X_train_combo2 = np.hstack([X_train_emb, X_train_morgan])
    X_test_combo2 = np.hstack([X_test_emb, X_test_morgan])

    run_experiment(
        "lgbm_chemberta+morgan_r2",
        X_train_combo2, y_train.values, X_test_combo2, test_df,
        "chemberta_384+morgan_r2_2048",
        "LightGBM with ChemBERTa embeddings + Morgan r2 FP",
    )

    # --- ChemBERTa + desc + Morgan (full combo) ---
    X_train_all = np.hstack([X_train_emb, X_train_desc, X_train_morgan])
    X_test_all = np.hstack([X_test_emb, X_test_desc, X_test_morgan])

    run_experiment(
        "lgbm_chemberta+desc+morgan_r2",
        X_train_all, y_train.values, X_test_all, test_df,
        "chemberta_384+rdkit_descriptors+morgan_r2_2048",
        "LightGBM with ChemBERTa + RDKit descriptors + Morgan r2",
    )

    print(f"\n{'=' * 60}")
    print("  Done! Check experiment_summary for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
