"""Run LightGBM experiments with all ChemBERTa embedding variants."""

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
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
)
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

EMBEDDING_TABLES = {
    "chemberta_77m_mlm": "compound_chemberta",
    "chemberta_77m_mtr": "compound_chemberta_mtr",
    "chemberta_100m_mlm": "compound_chemberta_100m",
    "chemberta_10m_mlm": "compound_chemberta_10m",
    "chemberta_10m_mtr": "compound_chemberta_10m_mtr",
    "chemberta_5m_mlm": "compound_chemberta_5m",
    "chemberta_5m_mtr": "compound_chemberta_5m_mtr",
    "chemberta_zinc_v1": "compound_chemberta_zinc_v1",
    "bert_base_smiles": "compound_bert_smiles",
}


def load_embeddings(table: str, compound_ids: list[int]) -> np.ndarray:
    """Load embeddings from a DB table for given compound IDs, preserving order."""
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
    """Load compound IDs for train or test split."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    if split == "train":
        cur.execute("SELECT compound_id FROM train_activity ORDER BY compound_id")
    else:
        cur.execute("SELECT compound_id FROM test_activity ORDER BY compound_id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


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

    submission = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
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
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, avg_rounds={avg_rounds}",
    )
    return oof_metrics


def main():
    print("Loading base data...")
    test_df = load_test_smiles()
    train_desc_df = load_train_descriptors()
    test_desc_df = load_test_descriptors()
    train_smiles_df = load_train_smiles_target()
    y_train = train_smiles_df["pec50"].values

    X_train_desc = train_desc_df[DESCRIPTOR_COLS].values
    X_test_desc = test_desc_df[DESCRIPTOR_COLS].values

    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")

    # Morgan FP (precompute once)
    train_mols = smiles_to_mols(train_smiles_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])
    X_train_morgan = FP_REGISTRY["morgan_r2_2048"](train_mols)
    X_test_morgan = FP_REGISTRY["morgan_r2_2048"](test_mols)

    results = {}

    for emb_name, table in EMBEDDING_TABLES.items():
        print(f"\nLoading {emb_name} from {table}...")
        X_train_emb = load_embeddings(table, train_ids)
        X_test_emb = load_embeddings(table, test_ids)
        dim = X_train_emb.shape[1]
        print(f"  dim={dim}")

        # Solo
        name = f"lgbm_{emb_name}"
        m = run_experiment(
            name,
            X_train_emb,
            y_train,
            X_test_emb,
            test_df,
            f"{emb_name}_{dim}d",
            f"LightGBM with {emb_name} embeddings ({dim}d)",
        )
        results[name] = m

        # + desc + morgan (best combo template)
        name = f"lgbm_{emb_name}+desc+morgan"
        X_train_all = np.hstack([X_train_emb, X_train_desc, X_train_morgan])
        X_test_all = np.hstack([X_test_emb, X_test_desc, X_test_morgan])
        m = run_experiment(
            name,
            X_train_all,
            y_train,
            X_test_all,
            test_df,
            f"{emb_name}_{dim}d+rdkit_desc+morgan_r2",
            f"LightGBM with {emb_name} + RDKit desc + Morgan r2",
        )
        results[name] = m

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY (sorted by RAE)")
    print(f"{'=' * 70}")
    for name, m in sorted(results.items(), key=lambda x: x[1]["RAE"]):
        print(
            f"  {name:<45} RAE={m['RAE']:.4f}  R2={m['R2']:.4f}  Spearman={m['Spearman_R']:.4f}"
        )


if __name__ == "__main__":
    main()
