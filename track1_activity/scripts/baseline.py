"""Baseline model for Track 1 Activity Prediction.

LightGBM with RDKit descriptors from PostgreSQL.
Evaluated with 5-fold CV using competition metrics.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.model_selection import KFold

OUTPUT_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}

DESCRIPTOR_COLS = [
    "amw", "exactmw", "logp", "tpsa", "labuteasa", "fractioncsp3",
    "hba", "hbd",
    "num_atoms", "num_heavy_atoms", "num_heteroatoms",
    "num_rotatable_bonds", "num_amide_bonds",
    "num_rings", "num_aromatic_rings", "num_aliphatic_rings",
    "num_aromatic_carbocycles", "num_aromatic_heterocycles",
    "num_aliphatic_carbocycles", "num_aliphatic_heterocycles",
    "num_saturated_rings", "num_saturated_carbocycles",
    "num_saturated_heterocycles", "num_heterocycles",
    "num_spiro_atoms", "num_bridgehead_atoms",
    "chi0v", "chi1v", "chi2v", "chi3v", "chi4v",
    "chi0n", "chi1n", "chi2n", "chi3n", "chi4n",
    "kappa1", "kappa2", "kappa3", "phi", "hallkieralpha",
]

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


def load_train_data():
    """Load train data with descriptors from DB."""
    conn = psycopg2.connect(**DB_PARAMS)
    query = """
        SELECT c.smiles, c.molecule_name, t.pec50,
               {desc_cols}
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        JOIN compound_descriptors d ON d.compound_id = c.id
    """.format(desc_cols=", ".join(f"d.{col}" for col in DESCRIPTOR_COLS))
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_test_data():
    """Load test data with descriptors from DB."""
    conn = psycopg2.connect(**DB_PARAMS)
    query = """
        SELECT c.smiles, c.molecule_name,
               {desc_cols}
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        JOIN compound_descriptors d ON d.compound_id = c.id
    """.format(desc_cols=", ".join(f"d.{col}" for col in DESCRIPTOR_COLS))
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute competition metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    # RAE = sum(|y_true - y_pred|) / sum(|y_true - mean(y_true)|)
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    spearman_r = stats.spearmanr(y_true, y_pred).statistic
    kendall_tau = stats.kendalltau(y_true, y_pred).statistic
    return {
        "MAE": mae,
        "RAE": rae,
        "R2": r2,
        "Spearman_R": spearman_r,
        "Kendall_Tau": kendall_tau,
    }


def print_metrics(metrics: dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
    print(
        f"  {prefix}RAE={metrics['RAE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"R2={metrics['R2']:.4f}  Spearman={metrics['Spearman_R']:.4f}  "
        f"Kendall={metrics['Kendall_Tau']:.4f}"
    )


def main():
    print("Loading data from DB...")
    train_df = load_train_data()
    test_df = load_test_data()
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train = train_df[DESCRIPTOR_COLS].values
    y_train = train_df["pec50"].values
    X_test = test_df[DESCRIPTOR_COLS].values

    # --- 5-Fold Cross Validation ---
    print("\n5-Fold Cross Validation:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train))
    fold_metrics = []

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
        print_metrics(metrics, label=f"Fold {fold}")

    # Overall OOF metrics
    print("\nOverall OOF:")
    oof_metrics = compute_metrics(y_train, oof_preds)
    print_metrics(oof_metrics)

    # Mean ± std across folds
    print("\nMean ± Std across folds:")
    for metric_name in ["RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau"]:
        values = [m[metric_name] for m in fold_metrics]
        print(f"  {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

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
    final_model = lgb.train(
        LGB_PARAMS,
        dtrain_full,
        num_boost_round=1000,
    )

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
