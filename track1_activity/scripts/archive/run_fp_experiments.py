"""Run LightGBM experiments with various fingerprints."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import load_train_smiles_target, load_test_smiles, load_train_descriptors, load_test_descriptors, DESCRIPTOR_COLS
from evaluate import compute_metrics, print_metrics, print_fold_summary, record_experiment
from features import FP_REGISTRY, smiles_to_mols

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

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


def run_single_experiment(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
    feature_set: str,
    description: str,
):
    """Run 5-fold CV, train final model, save submission, record to DB."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Features: {X_train.shape[1]}")
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

    # Overall OOF
    print("\n  Overall OOF:")
    oof_metrics = compute_metrics(y_train, oof_preds)
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model
    print("\n  Training final model...")
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    avg_rounds = int(np.mean(num_boost_rounds))
    final_model = lgb.train(LGB_PARAMS, dtrain_full, num_boost_round=avg_rounds)

    test_preds = final_model.predict(X_test)
    print(f"  Test preds: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    # Save submission
    submission = pd.DataFrame({
        "SMILES": test_df["smiles"],
        "Molecule Name": test_df["molecule_name"],
        "pEC50": test_preds,
    })
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path.name}")

    # Record to DB
    rel_path = f"track1_activity/submissions/{name}.csv"
    record_experiment(
        name=name,
        description=description,
        model_type="lightgbm",
        feature_set=feature_set,
        hyperparameters=LGB_PARAMS,
        fold_metrics=fold_metrics,
        submission_path=rel_path,
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, avg_rounds={avg_rounds}",
    )

    return oof_metrics


def main():
    # Load data
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_desc_df = load_train_descriptors()
    test_desc_df = load_test_descriptors()
    y_train = train_df["pec50"].values

    print("Computing molecular objects...")
    train_mols = smiles_to_mols(train_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])

    # Descriptor features (for combos)
    X_train_desc = train_desc_df[DESCRIPTOR_COLS].values
    X_test_desc = test_desc_df[DESCRIPTOR_COLS].values

    results = {}

    # --- Run each fingerprint type ---
    for fp_name, fp_func in FP_REGISTRY.items():
        exp_name = f"lgbm_{fp_name}"
        print(f"\nComputing {fp_name}...")
        X_train_fp = fp_func(train_mols)
        X_test_fp = fp_func(test_mols)

        metrics = run_single_experiment(
            name=exp_name,
            X_train=X_train_fp,
            y_train=y_train,
            X_test=X_test_fp,
            test_df=test_df,
            feature_set=fp_name,
            description=f"LightGBM with {fp_name} fingerprint",
        )
        results[exp_name] = metrics

    # --- Descriptor + Morgan combo ---
    for fp_name in ["morgan_r2_2048", "morgan_r3_2048"]:
        combo_name = f"lgbm_desc+{fp_name}"
        print(f"\nComputing combo: descriptors + {fp_name}...")
        X_train_fp = FP_REGISTRY[fp_name](train_mols)
        X_test_fp = FP_REGISTRY[fp_name](test_mols)
        X_train_combo = np.hstack([X_train_desc, X_train_fp])
        X_test_combo = np.hstack([X_test_desc, X_test_fp])

        metrics = run_single_experiment(
            name=combo_name,
            X_train=X_train_combo,
            y_train=y_train,
            X_test=X_test_combo,
            test_df=test_df,
            feature_set=f"rdkit_descriptors+{fp_name}",
            description=f"LightGBM with RDKit descriptors + {fp_name}",
        )
        results[combo_name] = metrics

    # --- Descriptor + MACCS combo ---
    combo_name = "lgbm_desc+maccs"
    print(f"\nComputing combo: descriptors + MACCS...")
    X_train_maccs = FP_REGISTRY["maccs"](train_mols)
    X_test_maccs = FP_REGISTRY["maccs"](test_mols)
    X_train_combo = np.hstack([X_train_desc, X_train_maccs])
    X_test_combo = np.hstack([X_test_desc, X_test_maccs])

    metrics = run_single_experiment(
        name=combo_name,
        X_train=X_train_combo,
        y_train=y_train,
        X_test=X_test_combo,
        test_df=test_df,
        feature_set="rdkit_descriptors+maccs",
        description="LightGBM with RDKit descriptors + MACCS keys",
    )
    results[combo_name] = metrics

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY (sorted by RAE)")
    print(f"{'=' * 60}")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["RAE"])
    for name, m in sorted_results:
        print(f"  {name:<35} RAE={m['RAE']:.4f}  R2={m['R2']:.4f}  Spearman={m['Spearman_R']:.4f}")


if __name__ == "__main__":
    main()
