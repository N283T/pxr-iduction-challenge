"""Run LightGBM experiments with Mordred descriptors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.model_selection import KFold

from data import load_train_smiles_target, load_test_smiles, load_train_descriptors, load_test_descriptors, DESCRIPTOR_COLS
from evaluate import compute_metrics, print_metrics, print_fold_summary, record_experiment
from features import smiles_to_mols

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


def compute_mordred(mols: list) -> tuple[np.ndarray, list[str]]:
    """Compute Mordred 2D descriptors, return (array, col_names)."""
    calc = Calculator(descriptors, ignore_3D=True)
    print(f"  Computing {len(calc.descriptors)} Mordred descriptors for {len(mols)} molecules...")
    df = calc.pandas(mols, quiet=True)

    # Convert to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns that are all NaN or have zero variance
    df = df.dropna(axis=1, how="all")
    nunique = df.nunique()
    df = df.loc[:, nunique > 1]

    # Fill remaining NaN with column median
    df = df.fillna(df.median())

    # Replace inf with NaN then fill
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    col_names = list(df.columns)
    print(f"  After cleanup: {len(col_names)} features")
    return df.values.astype(np.float32), col_names


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
            LGB_PARAMS, dtrain, num_boost_round=2000,
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

    # Final model
    avg_rounds = int(np.mean(num_boost_rounds))
    final_model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, label=y_train), num_boost_round=avg_rounds)
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
        name=name, description=description, model_type="lightgbm",
        feature_set=feature_set, hyperparameters=LGB_PARAMS,
        fold_metrics=fold_metrics, submission_path=f"track1_activity/submissions/{name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, avg_rounds={avg_rounds}",
    )
    return oof_metrics, final_model


def main():
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_desc_df = load_train_descriptors()
    test_desc_df = load_test_descriptors()
    y_train = train_df["pec50"].values

    print("Computing molecules...")
    train_mols = smiles_to_mols(train_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])

    # --- Mordred only ---
    print("\nComputing Mordred descriptors (train)...")
    X_train_mordred, mordred_cols = compute_mordred(train_mols)
    print("Computing Mordred descriptors (test)...")
    X_test_mordred, _ = compute_mordred(test_mols)

    # Align columns (test might have different cleanup results)
    # Use train columns as reference
    if X_test_mordred.shape[1] != X_train_mordred.shape[1]:
        print(f"  WARNING: train has {X_train_mordred.shape[1]} cols, test has {X_test_mordred.shape[1]} cols")
        # Recompute test with same calculator to ensure alignment
        calc = Calculator(descriptors, ignore_3D=True)
        test_mordred_df = calc.pandas(test_mols, quiet=True)
        for col in test_mordred_df.columns:
            test_mordred_df[col] = pd.to_numeric(test_mordred_df[col], errors="coerce")
        test_mordred_df = test_mordred_df[mordred_cols]
        test_mordred_df = test_mordred_df.fillna(test_mordred_df.median())
        test_mordred_df = test_mordred_df.replace([np.inf, -np.inf], np.nan).fillna(test_mordred_df.median())
        X_test_mordred = test_mordred_df.values.astype(np.float32)

    run_experiment(
        "lgbm_mordred", X_train_mordred, y_train, X_test_mordred, test_df,
        "mordred_2d", "LightGBM with Mordred 2D descriptors",
    )

    # --- Mordred + RDKit descriptors ---
    X_train_rdkit = train_desc_df[DESCRIPTOR_COLS].values
    X_test_rdkit = test_desc_df[DESCRIPTOR_COLS].values
    X_train_combo = np.hstack([X_train_rdkit, X_train_mordred])
    X_test_combo = np.hstack([X_test_rdkit, X_test_mordred])

    run_experiment(
        "lgbm_rdkit_desc+mordred", X_train_combo, y_train, X_test_combo, test_df,
        "rdkit_descriptors+mordred_2d", "LightGBM with RDKit descriptors + Mordred 2D",
    )

    # --- Mordred + Morgan FP ---
    from features import FP_REGISTRY
    X_train_morgan = FP_REGISTRY["morgan_r2_2048"](train_mols)
    X_test_morgan = FP_REGISTRY["morgan_r2_2048"](test_mols)
    X_train_combo2 = np.hstack([X_train_mordred, X_train_morgan])
    X_test_combo2 = np.hstack([X_test_mordred, X_test_morgan])

    run_experiment(
        "lgbm_mordred+morgan_r2", X_train_combo2, y_train, X_test_combo2, test_df,
        "mordred_2d+morgan_r2_2048", "LightGBM with Mordred 2D + Morgan r2 FP",
    )

    # --- Mordred + RDKit desc + Morgan FP (kitchen sink) ---
    X_train_all = np.hstack([X_train_rdkit, X_train_mordred, X_train_morgan])
    X_test_all = np.hstack([X_test_rdkit, X_test_mordred, X_test_morgan])

    run_experiment(
        "lgbm_all_features", X_train_all, y_train, X_test_all, test_df,
        "rdkit_desc+mordred_2d+morgan_r2_2048", "LightGBM with all features combined",
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("  Done! Check experiment_summary for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
