"""Re-evaluate top models with Murcko scaffold split CV.

Compares random vs scaffold split to assess CV optimism.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
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


def run_cv(name, X_train, y_train, X_test, test_df, feature_set, description, splits):
    """Run CV with given splits, train final model, save submission, record to DB."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({X_train.shape[1]} features)")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    num_boost_rounds = []

    for fold, (train_idx, val_idx) in enumerate(splits):
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

    # Final model on all data
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

    record_experiment(
        name=name,
        description=description,
        model_type="lightgbm",
        feature_set=feature_set,
        hyperparameters=LGB_PARAMS,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, scaffold_split, avg_rounds={avg_rounds}",
    )
    return oof_metrics, oof_preds


def main():
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_desc_df = load_train_descriptors()
    test_desc_df = load_test_descriptors()
    y_train = train_df["pec50"].values

    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")

    # Pre-compute features
    print("Computing features...")
    train_mols = smiles_to_mols(train_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])

    X_desc_tr = train_desc_df[DESCRIPTOR_COLS].values
    X_desc_te = test_desc_df[DESCRIPTOR_COLS].values
    X_morgan_tr = FP_REGISTRY["morgan_r2_2048"](train_mols)
    X_morgan_te = FP_REGISTRY["morgan_r2_2048"](test_mols)
    X_chemeleon_tr = load_embeddings("compound_chemeleon", train_ids)
    X_chemeleon_te = load_embeddings("compound_chemeleon", test_ids)

    # Scaffold split
    print("Computing scaffold splits...")
    scaffold_splits = scaffold_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)
    random_splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(y_train))

    # Define feature configurations to test
    configs = {
        "desc+morgan_r2": {
            "X_train": np.hstack([X_desc_tr, X_morgan_tr]),
            "X_test": np.hstack([X_desc_te, X_morgan_te]),
            "feature_set": "rdkit_descriptors+morgan_r2_2048",
        },
        "chemeleon+desc+morgan": {
            "X_train": np.hstack([X_chemeleon_tr, X_desc_tr, X_morgan_tr]),
            "X_test": np.hstack([X_chemeleon_te, X_desc_te, X_morgan_te]),
            "feature_set": "chemeleon_300d+rdkit_descriptors+morgan_r2_2048",
        },
    }

    # Also load mordred+morgan
    from mordred import Calculator, descriptors as mordred_descs

    print("Computing Mordred descriptors...")
    calc = Calculator(mordred_descs, ignore_3D=True)
    train_mordred_df = calc.pandas(train_mols, quiet=True)
    for col in train_mordred_df.columns:
        train_mordred_df[col] = pd.to_numeric(train_mordred_df[col], errors="coerce")
    train_mordred_df = train_mordred_df.dropna(axis=1, how="all")
    train_mordred_df = train_mordred_df.loc[:, train_mordred_df.nunique() > 1]
    train_mordred_df = train_mordred_df.replace([np.inf, -np.inf], np.nan)
    mordred_medians = train_mordred_df.median()
    mordred_cols = list(train_mordred_df.columns)
    train_mordred_df = train_mordred_df.fillna(mordred_medians)
    X_mordred_tr = train_mordred_df.values.astype(np.float32)

    test_mordred_df = calc.pandas(test_mols, quiet=True)
    for col in test_mordred_df.columns:
        test_mordred_df[col] = pd.to_numeric(test_mordred_df[col], errors="coerce")
    test_mordred_df = test_mordred_df.reindex(columns=mordred_cols)
    test_mordred_df = test_mordred_df.replace([np.inf, -np.inf], np.nan)
    test_mordred_df = test_mordred_df.fillna(mordred_medians[mordred_cols])
    X_mordred_te = test_mordred_df.values.astype(np.float32)

    configs["mordred+morgan_r2"] = {
        "X_train": np.hstack([X_mordred_tr, X_morgan_tr]),
        "X_test": np.hstack([X_mordred_te, X_morgan_te]),
        "feature_set": "mordred_2d+morgan_r2_2048",
    }

    # Run experiments
    results = {}

    for config_name, config in configs.items():
        # Scaffold split
        name = f"scaffold_{config_name}"
        m, oof = run_cv(
            name,
            config["X_train"], y_train, config["X_test"], test_df,
            config["feature_set"],
            f"LightGBM {config_name} (scaffold split CV)",
            scaffold_splits,
        )
        results[name] = m

        # Random split for comparison
        name = f"random_{config_name}"
        m, oof = run_cv(
            name,
            config["X_train"], y_train, config["X_test"], test_df,
            config["feature_set"],
            f"LightGBM {config_name} (random split CV)",
            random_splits,
        )
        results[name] = m

    # Summary
    print(f"\n{'=' * 70}")
    print("  COMPARISON: Scaffold vs Random Split")
    print(f"{'=' * 70}")
    for config_name in configs:
        s = results[f"scaffold_{config_name}"]
        r = results[f"random_{config_name}"]
        delta = s["RAE"] - r["RAE"]
        print(f"\n  {config_name}:")
        print(f"    Random:   RAE={r['RAE']:.4f}  R2={r['R2']:.4f}  Spearman={r['Spearman_R']:.4f}")
        print(f"    Scaffold: RAE={s['RAE']:.4f}  R2={s['R2']:.4f}  Spearman={s['Spearman_R']:.4f}")
        print(f"    Delta:    RAE={delta:+.4f} ({'worse' if delta > 0 else 'better'} with scaffold)")


if __name__ == "__main__":
    main()
