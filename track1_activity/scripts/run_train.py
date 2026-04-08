#!/usr/bin/env -S pixi run python
"""Unified training script for single model+feature experiments.

Usage:
    pixi run python run_train.py --model lgbm --feature mordred
    pixi run python run_train.py --model xgboost --feature morgan_r2_2048 --split umap
    pixi run python run_train.py --model catboost --feature count_morgan_r2_2048 --trials 0

Key changes from legacy scripts:
- No inner CV: Optuna tunes directly on outer fold's val set
- No final re-tuning: reuses median best params from CV folds
- Unified model/feature/split selection via CLI args
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import pandas as pd
import psycopg2

from data import (
    DB_PARAMS,
    DESCRIPTOR_COLS,
    JAZZY_FEATURE_COLS,
    get_conn,
    load_jazzy,
    load_mordred,
    load_test_descriptors,
    load_test_smiles,
    load_train_descriptors,
    load_train_mordred,
    load_train_smiles_target,
)
from evaluate import (
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from features import FP_REGISTRY, smiles_to_mols
from pseudo_labels import (
    augment_fold,
    build_pseudo_feature_matrix,
    load_pseudo_labels,
)
from splits import scaffold_split_indices, umap_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Embedding tables in DB
EMBEDDING_TABLES = {
    "chemeleon": "compound_chemeleon",
    "chemberta_77m_mlm": "compound_chemberta",
    "chemberta_77m_mtr": "compound_chemberta_mtr",
    "chemberta_100m_mlm": "compound_chemberta_100m",
    "chemberta_10m_mlm": "compound_chemberta_10m",
    "chemberta_5m_mtr": "compound_chemberta_5m_mtr",
    "chemberta_zinc_v1": "compound_chemberta_zinc_v1",
    "bert_base_smiles": "compound_bert_smiles",
    "molformer_xl": "compound_molformer",
}

# Default hyperparams per model type (used when --trials 0)
DEFAULT_PARAMS = {
    "lgbm": {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 42,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.01,
        "lambda_l2": 1.0,
    },
    "xgboost": {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "gamma": 0.01,
    },
    "catboost": {
        "iterations": 2000,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "border_count": 128,
    },
}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_compound_ids(split: str) -> list[int]:
    conn = get_conn()
    cur = conn.cursor()
    table = "train_activity" if split == "train" else "test_activity"
    cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


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
    missing = set(compound_ids) - set(rows)
    if missing:
        raise ValueError(
            f"Embeddings missing for {len(missing)} compounds in {table}: "
            f"{sorted(missing)[:5]}..."
        )
    return np.array([rows[cid] for cid in compound_ids])


def load_features(feature_name: str, train_df, test_df):
    """Load feature matrices for train and test."""
    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")

    if feature_name == "rdkit_desc":
        train_desc = load_train_descriptors()
        test_desc = load_test_descriptors()
        return train_desc[DESCRIPTOR_COLS].values, test_desc[DESCRIPTOR_COLS].values

    if feature_name in ("mordred", "mordred_singleconc", "mordred_jazzy"):
        mordred_train, _ = load_train_mordred()
        mordred_test = load_mordred(test_ids)
        missing_train = set(train_ids) - set(mordred_train.index)
        missing_test = set(test_ids) - set(mordred_test.index)
        if missing_train:
            raise ValueError(
                f"Mordred missing for {len(missing_train)} train compounds"
            )
        if missing_test:
            raise ValueError(f"Mordred missing for {len(missing_test)} test compounds")
        common_cols = mordred_train.columns.intersection(mordred_test.columns)
        X_train = mordred_train.loc[train_ids, common_cols].values.astype(np.float32)
        X_test = mordred_test.loc[test_ids, common_cols].values.astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        if feature_name == "mordred_jazzy":
            jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
            jazzy_test = load_jazzy(test_ids).reindex(index=test_ids)
            missing_j_tr = int(jazzy_train.isna().any(axis=1).sum())
            missing_j_te = int(jazzy_test.isna().any(axis=1).sum())
            if missing_j_tr or missing_j_te:
                raise ValueError(
                    f"Jazzy missing rows: train={missing_j_tr}, test={missing_j_te}"
                )
            jt_tr = jazzy_train[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
            jt_te = jazzy_test[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
            X_train = np.concatenate([X_train, jt_tr], axis=1)
            X_test = np.concatenate([X_test, jt_te], axis=1)
            print(
                f"  jazzy concat: +{jt_tr.shape[1]} cols (train+test both fully populated)"
            )

        if feature_name == "mordred_singleconc":
            from data import load_singleconc_features

            sc_train = (
                load_singleconc_features(train_ids)
                .loc[train_ids]
                .to_numpy(dtype=np.float32)
            )
            sc_test = (
                load_singleconc_features(test_ids)
                .loc[test_ids]
                .to_numpy(dtype=np.float32)
            )
            # NaN preserved (LightGBM handles missing natively)
            X_train = np.concatenate([X_train, sc_train], axis=1)
            X_test = np.concatenate([X_test, sc_test], axis=1)
            n_sc_train_with_data = (~np.isnan(sc_train).all(axis=1)).sum()
            print(
                f"  single_conc concat: +{sc_train.shape[1]} cols, "
                f"{n_sc_train_with_data}/{len(train_ids)} train compounds have data"
            )

        return X_train, X_test

    if feature_name in EMBEDDING_TABLES:
        table = EMBEDDING_TABLES[feature_name]
        return load_embeddings(table, train_ids), load_embeddings(table, test_ids)

    if feature_name in FP_REGISTRY:
        train_mols = smiles_to_mols(train_df["smiles"])
        test_mols = smiles_to_mols(test_df["smiles"])
        return (
            FP_REGISTRY[feature_name](train_mols).astype(np.float32),
            FP_REGISTRY[feature_name](test_mols).astype(np.float32),
        )

    raise ValueError(
        f"Unknown feature: {feature_name}. "
        f"Available: rdkit_desc, mordred, {list(EMBEDDING_TABLES)}, {list(FP_REGISTRY)}"
    )


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------


def optuna_search_space(model_type: str, trial):
    """Generate hyperparameter candidates for Optuna trial."""
    if model_type == "lgbm":
        return {
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
    if model_type == "xgboost":
        return {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "seed": 42,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
    if model_type == "catboost":
        return {
            "iterations": 2000,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "verbose": 0,
            "random_seed": 42,
            "allow_writing_files": False,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 10.0, log=True
            ),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
    raise ValueError(f"Unknown model: {model_type}")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_predict(model_type: str, params: dict, X_tr, y_tr, X_va, y_va, w_tr=None):
    """Train a model and return (val_preds, best_iteration)."""
    if model_type == "lgbm":
        import lightgbm as lgb

        dt = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)
        model = lgb.train(
            params,
            dt,
            num_boost_round=2000,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return model.predict(X_va), model.best_iteration, model

    if w_tr is not None and model_type != "lgbm":
        raise NotImplementedError(
            f"Sample weights only supported for lgbm, got {model_type}"
        )

    if model_type == "xgboost":
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        return model.predict(dval), model.best_iteration, model

    if model_type == "catboost":
        import catboost as cb

        model = cb.CatBoostRegressor(**params)
        model.fit(
            cb.Pool(X_tr, label=y_tr),
            eval_set=cb.Pool(X_va, label=y_va),
            early_stopping_rounds=50,
        )
        return model.predict(X_va), model.best_iteration_, model

    raise ValueError(f"Unknown model: {model_type}")


def train_final(
    model_type: str,
    params: dict,
    X_train,
    y_train,
    num_rounds: int,
    sample_weight=None,
):
    """Train final model on all training data.

    ``sample_weight`` is supported for LightGBM only; XGBoost/CatBoost raise
    NotImplementedError when a non-None weight is supplied (matching
    ``train_predict``).
    """
    if model_type == "lgbm":
        import lightgbm as lgb

        return lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train, weight=sample_weight),
            num_boost_round=num_rounds,
        )

    if sample_weight is not None:
        raise NotImplementedError(
            f"Sample weights only supported for lgbm, got {model_type}"
        )

    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.train(
            params, xgb.DMatrix(X_train, label=y_train), num_boost_round=num_rounds
        )

    if model_type == "catboost":
        import catboost as cb

        p = {**params, "iterations": num_rounds}
        model = cb.CatBoostRegressor(**p)
        model.fit(cb.Pool(X_train, label=y_train))
        return model

    raise ValueError(f"Unknown model: {model_type}")


def predict_final(model_type: str, model, X_test):
    """Predict with final model."""
    if model_type == "xgboost":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X_test))
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run(args):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(
        f"Model: {args.model}, Feature: {args.feature}, Split: {args.split}, Trials: {args.trials}"
    )

    # Load data
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].values

    X_train, X_test = load_features(args.feature, train_df, test_df)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Optional pseudo-labels
    pseudo_X = None
    pseudo_y = None
    pseudo_w = None
    pseudo_count = 0
    if args.pseudo:
        if args.trials > 0:
            raise ValueError(
                "--pseudo requires --trials 0 (Optuna+pseudo not supported)"
            )
        print(f"Loading pseudo-labels from {args.pseudo}...")
        pseudo_df = load_pseudo_labels(Path(args.pseudo))
        if args.pseudo_min_confidence > 0:
            n_before = len(pseudo_df)
            pseudo_df = pseudo_df[
                pseudo_df["confidence"] >= args.pseudo_min_confidence
            ].reset_index(drop=True)
            print(
                f"  confidence filter ≥{args.pseudo_min_confidence}: "
                f"{n_before} → {len(pseudo_df)} pseudo compounds"
            )
        train_ids = set(load_compound_ids("train"))
        pseudo_ids = set(pseudo_df["compound_id"].astype(int))
        overlap = train_ids & pseudo_ids
        assert not overlap, (
            f"Pseudo compound_ids overlap with train ({len(overlap)} ids); "
            f"fold-safety violated"
        )
        if args.feature != "mordred":
            raise NotImplementedError(
                f"--pseudo currently supports --feature mordred only, "
                f"got {args.feature!r}"
            )
        # Canonical column order: same intersection load_features built above.
        mordred_train_df, _ = load_train_mordred()
        mordred_test_df = load_mordred(load_compound_ids("test"))
        feature_columns = list(
            mordred_train_df.columns.intersection(mordred_test_df.columns)
        )
        assert X_train.shape[1] == len(feature_columns), (
            f"Train feature column count mismatch: X_train has {X_train.shape[1]} "
            f"cols, expected {len(feature_columns)}"
        )
        pseudo_X = build_pseudo_feature_matrix(
            args.feature, pseudo_df, feature_columns=feature_columns
        )
        assert pseudo_X.shape[1] == X_train.shape[1], (
            f"Pseudo/train column mismatch: pseudo={pseudo_X.shape[1]}, "
            f"train={X_train.shape[1]}"
        )
        pseudo_y = pseudo_df["pseudo_pec50"].values.astype(np.float32)
        pseudo_w = (
            pseudo_df["confidence"].values.astype(np.float32) * args.pseudo_weight
        )
        pseudo_count = len(pseudo_df)
        print(
            f"  pseudo_X: {pseudo_X.shape}, weight={args.pseudo_weight}, "
            f"effective sum={pseudo_w.sum():.1f}"
        )

    # CV splits
    smiles_list = train_df["smiles"].tolist()
    if args.split == "scaffold":
        outer_splits = scaffold_split_indices(smiles_list, n_splits=5, seed=42)
    elif args.split == "umap":
        outer_splits = umap_split_indices(
            smiles_list, n_splits=5, n_clusters=50, seed=42
        )
    else:
        raise ValueError(f"Unknown split: {args.split}")

    # Experiment name
    exp_name = f"{args.model}_{args.feature}_{args.split}"
    if args.trials == 0:
        exp_name += "_default"
    if args.gap_lambda > 0:
        exp_name += f"_gap{args.gap_lambda}"
    if args.pseudo:
        exp_name += f"_pseudo{args.pseudo_weight}"
        if args.pseudo_min_confidence > 0:
            exp_name += f"_minc{args.pseudo_min_confidence}"
    print(f"  Experiment: {exp_name}")

    # Training loop
    oof_preds = np.zeros(len(y_train))
    fold_metrics = []
    fold_best_params = []
    num_boost_rounds = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr = None

        if pseudo_X is not None:
            real_n = len(X_tr)
            X_tr, y_tr, w_tr = augment_fold(
                X_tr, y_tr, pseudo_X, pseudo_y, pseudo_w, base_weight=1.0
            )
            print(
                f"  [Fold {fold}] augmented: real={real_n} + pseudo={pseudo_count} "
                f"= {len(X_tr)} train, val={len(X_va)} (real only)"
            )

        if args.trials > 0:
            # Optuna tuning directly on outer fold val (no inner CV)
            gap_lambda = args.gap_lambda
            print(
                f"  [Fold {fold}] Tuning ({args.trials} trials, gap_λ={gap_lambda})..."
            )

            def objective(trial):
                params = optuna_search_space(args.model, trial)
                val_preds, _, model = train_predict(
                    args.model, params, X_tr, y_tr, X_va, y_va
                )
                val_mae = np.mean(np.abs(y_va - val_preds))
                if gap_lambda > 0:
                    train_preds = predict_final(args.model, model, X_tr)
                    train_mae = np.mean(np.abs(y_tr - train_preds))
                    gap = abs(train_mae - val_mae)
                    return val_mae + gap_lambda * gap
                return val_mae

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.trials)
            best_params = optuna_search_space(args.model, study.best_trial)
        else:
            print(f"  [Fold {fold}] Using default params...")
            best_params = DEFAULT_PARAMS[args.model].copy()

        fold_best_params.append(best_params)

        # Train with best params
        vp, best_iter, _ = train_predict(
            args.model, best_params, X_tr, y_tr, X_va, y_va, w_tr=w_tr
        )
        oof_preds[va_idx] = vp
        metrics = compute_metrics(y_va, vp)
        fold_metrics.append(metrics)
        num_boost_rounds.append(best_iter)
        print_metrics(metrics, label=f"Fold {fold}")

    # OOF summary
    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model: use median best_iteration, best fold's params
    avg_rounds = int(np.median(num_boost_rounds))
    best_fold = int(np.argmin([m["RAE"] for m in fold_metrics]))
    final_params = fold_best_params[best_fold]
    print(
        f"\n  Final model: {avg_rounds} rounds, params from fold {best_fold} "
        f"(RAE={fold_metrics[best_fold]['RAE']:.4f})"
    )

    if pseudo_X is not None:
        X_final = np.concatenate([X_train, pseudo_X], axis=0)
        y_final = np.concatenate([y_train, pseudo_y], axis=0)
        w_final = np.concatenate(
            [np.ones(len(X_train), dtype=np.float32), pseudo_w.astype(np.float32)],
            axis=0,
        )
        final_model = train_final(
            args.model,
            final_params,
            X_final,
            y_final,
            avg_rounds,
            sample_weight=w_final,
        )
    else:
        final_model = train_final(
            args.model, final_params, X_train, y_train, avg_rounds
        )
    test_preds = predict_final(args.model, final_model, X_test)
    print(f"  Test preds: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    # Save submission
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    # Record to DB
    exp_id = record_experiment(
        name=exp_name,
        description=f"{args.model} + {args.feature} ({args.split} split)",
        model_type=args.model,
        feature_set=args.feature,
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, {args.split}_split, "
            f"{args.trials}trials, gap_lambda={args.gap_lambda}"
            + (
                f", pseudo={args.pseudo} weight={args.pseudo_weight} n={pseudo_count}"
                if args.pseudo
                else ""
            )
        ),
    )
    save_oof_predictions(exp_id, oof_preds)

    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")
    return oof_metrics


def main():
    all_features = (
        ["rdkit_desc", "mordred", "mordred_singleconc", "mordred_jazzy"]
        + list(FP_REGISTRY.keys())
        + list(EMBEDDING_TABLES.keys())
    )

    parser = argparse.ArgumentParser(description="Unified model training")
    parser.add_argument(
        "--model", choices=["lgbm", "xgboost", "catboost"], required=True
    )
    parser.add_argument("--feature", choices=all_features, required=True)
    parser.add_argument("--split", choices=["scaffold", "umap"], default="scaffold")
    parser.add_argument(
        "--trials", type=int, default=20, help="Optuna trials (0=default params)"
    )
    parser.add_argument(
        "--gap-lambda",
        type=float,
        default=0.0,
        help="Gap regularization lambda (0=off, 0.5-2.0 recommended)",
    )
    parser.add_argument(
        "--pseudo",
        type=str,
        default=None,
        help="Path to pseudo-label parquet (e.g. data/pseudo_labels.parquet)",
    )
    parser.add_argument(
        "--pseudo-weight",
        type=float,
        default=0.3,
        help="Weight multiplier for pseudo samples (multiplied by per-row confidence)",
    )
    parser.add_argument(
        "--pseudo-min-confidence",
        type=float,
        default=0.0,
        help="Drop pseudo samples with confidence below this threshold (0 = keep all)",
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
