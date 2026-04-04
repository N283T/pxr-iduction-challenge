"""Baseline model for Track 1 Activity Prediction.

LightGBM with RDKit descriptors from PostgreSQL.
Evaluated with 5-fold CV using competition metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import DESCRIPTOR_COLS, load_train_descriptors, load_test_descriptors
from evaluate import compute_metrics, print_metrics, print_fold_summary, record_experiment

OUTPUT_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")

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


def main():
    print("Loading data from DB...")
    train_df = load_train_descriptors()
    test_df = load_test_descriptors()
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train = train_df[DESCRIPTOR_COLS].values
    y_train = train_df["pec50"].values
    X_test = test_df[DESCRIPTOR_COLS].values

    # --- 5-Fold Cross Validation ---
    print("\n5-Fold Cross Validation:")
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
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        metrics = compute_metrics(y_val, val_pred)
        fold_metrics.append(metrics)
        num_boost_rounds.append(model.best_iteration)
        print_metrics(metrics, label=f"Fold {fold}")

    # Overall OOF metrics
    print("\nOverall OOF:")
    oof_metrics = compute_metrics(y_train, oof_preds)
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # --- Feature Importance ---
    print("\nTop 15 Feature Importance (last fold):")
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(
        zip(DESCRIPTOR_COLS, importance), key=lambda x: x[1], reverse=True
    )
    for name, imp in feat_imp[:15]:
        print(f"  {name:<30} {imp:.1f}")

    # --- Train final model on all data and predict test ---
    print("\nTraining final model on all data...")
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(LGB_PARAMS, dtrain_full, num_boost_round=1000)

    test_preds = final_model.predict(X_test)
    print(f"  Test predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")
    print(f"  Train target:     mean={y_train.mean():.3f}, std={y_train.std():.3f}")

    # --- Generate submission ---
    submission = pd.DataFrame({
        "SMILES": test_df["smiles"],
        "Molecule Name": test_df["molecule_name"],
        "pEC50": test_preds,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = OUTPUT_DIR.joinpath("baseline_lgbm_descriptors.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(f"  Rows: {len(submission)}, Columns: {list(submission.columns)}")


if __name__ == "__main__":
    main()
