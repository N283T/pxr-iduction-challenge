"""Optuna-tuned LightGBM for each individual feature source.

Each model uses a single feature type, tuned independently with nested CV.
These diverse models are candidates for ensemble.
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

N_OPTUNA_TRIALS = 50
INNER_CV_FOLDS = 3

EMBEDDING_TABLES = {
    "chemeleon": "compound_chemeleon",
    "chemberta_77m_mlm": "compound_chemberta",
    "chemberta_77m_mtr": "compound_chemberta_mtr",
    "chemberta_100m_mlm": "compound_chemberta_100m",
    "chemberta_10m_mlm": "compound_chemberta_10m",
    "chemberta_5m_mtr": "compound_chemberta_5m_mtr",
    "chemberta_zinc_v1": "compound_chemberta_zinc_v1",
    "bert_base_smiles": "compound_bert_smiles",
}


def load_embeddings(table: str, compound_ids: list[int]) -> np.ndarray:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    ph = ",".join(["%s"] * len(compound_ids))
    cur.execute(
        f"SELECT compound_id, embedding FROM {table} WHERE compound_id IN ({ph})",
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


def create_objective(X_fold, y_fold):
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
        kf = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=42)
        scores = []
        for ti, vi in kf.split(X_fold):
            dt = lgb.Dataset(X_fold[ti], label=y_fold[ti])
            dv = lgb.Dataset(X_fold[vi], label=y_fold[vi], reference=dt)
            m = lgb.train(
                params,
                dt,
                num_boost_round=2000,
                valid_sets=[dv],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            scores.append(np.mean(np.abs(y_fold[vi] - m.predict(X_fold[vi]))))
        return np.mean(scores)

    return objective


def run_single_optuna(
    name, X_train, y_train, X_test, test_df, feature_set, outer_splits
):
    """Nested CV: Optuna inner, scaffold outer."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({X_train.shape[1]} features)")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    num_boost_rounds = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        print(f"  [Fold {fold}] Tuning...")
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

        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)
        model = lgb.train(
            best_params,
            dt,
            num_boost_round=2000,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        vp = model.predict(X_va)
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
    print("  Tuning final model...")
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
    print(f"  Test preds: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    submission = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)

    exp_id = record_experiment(
        name=name,
        description=f"LightGBM Optuna single-feature ({feature_set})",
        model_type="lightgbm_optuna",
        feature_set=feature_set,
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, single_feature, scaffold_split, nested_cv",
    )
    save_oof_predictions(exp_id, oof_preds)
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

    train_mols = smiles_to_mols(train_df["smiles"])
    test_mols = smiles_to_mols(test_df["smiles"])

    # Scaffold splits
    outer_splits = scaffold_split_indices(
        train_df["smiles"].tolist(), n_splits=5, seed=42
    )

    # Build all single-feature configs
    configs = {}

    # RDKit descriptors
    configs["single_rdkit_desc"] = {
        "X_train": train_desc_df[DESCRIPTOR_COLS].values,
        "X_test": test_desc_df[DESCRIPTOR_COLS].values,
        "feature_set": "rdkit_descriptors",
    }

    # Fingerprints
    for fp_name in [
        "morgan_r2_2048",
        "morgan_r3_2048",
        "maccs",
        "avalon_2048",
        "atompair_2048",
    ]:
        X_tr = FP_REGISTRY[fp_name](train_mols)
        X_te = FP_REGISTRY[fp_name](test_mols)
        configs[f"single_{fp_name}"] = {
            "X_train": X_tr,
            "X_test": X_te,
            "feature_set": fp_name,
        }

    # Mordred
    print("Computing Mordred descriptors...")
    calc = Calculator(mordred_descs, ignore_3D=True)
    tr_mord = calc.pandas(train_mols, quiet=True)
    for col in tr_mord.columns:
        tr_mord[col] = pd.to_numeric(tr_mord[col], errors="coerce")
    tr_mord = tr_mord.dropna(axis=1, how="all")
    tr_mord = tr_mord.loc[:, tr_mord.nunique() > 1]
    tr_mord = tr_mord.replace([np.inf, -np.inf], np.nan)
    mord_medians = tr_mord.median()
    mord_cols = list(tr_mord.columns)
    tr_mord = tr_mord.fillna(mord_medians)

    te_mord = calc.pandas(test_mols, quiet=True)
    for col in te_mord.columns:
        te_mord[col] = pd.to_numeric(te_mord[col], errors="coerce")
    te_mord = te_mord.reindex(columns=mord_cols)
    te_mord = te_mord.replace([np.inf, -np.inf], np.nan)
    te_mord = te_mord.fillna(mord_medians[mord_cols])

    configs["single_mordred"] = {
        "X_train": tr_mord.values.astype(np.float32),
        "X_test": te_mord.values.astype(np.float32),
        "feature_set": "mordred_2d",
    }

    # Embeddings from DB
    for emb_name, table in EMBEDDING_TABLES.items():
        X_tr = load_embeddings(table, train_ids)
        X_te = load_embeddings(table, test_ids)
        configs[f"single_{emb_name}"] = {
            "X_train": X_tr,
            "X_test": X_te,
            "feature_set": emb_name,
        }

    print(f"\nTotal configs: {len(configs)}")
    for name, cfg in configs.items():
        print(f"  {name}: {cfg['X_train'].shape[1]} features")

    # Run all
    results = {}
    for config_name, config in configs.items():
        m, oof = run_single_optuna(
            config_name,
            config["X_train"],
            y_train,
            config["X_test"],
            test_df,
            config["feature_set"],
            outer_splits,
        )
        results[config_name] = m

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Single-Feature Optuna Models (scaffold split)")
    print(f"{'=' * 70}")
    for name, m in sorted(results.items(), key=lambda x: x[1]["RAE"]):
        print(
            f"  {name:<35} RAE={m['RAE']:.4f}  R2={m['R2']:.4f}  Spearman={m['Spearman_R']:.4f}"
        )


if __name__ == "__main__":
    main()
