#!/usr/bin/env python
"""Optuna tune LightGBM on mordred + boltz2_tier0 + boltz2_tier1 (1594 feat).

Search space matches existing LGBM tuning conventions (see run_train.py
``optuna_search_space`` for reference). 5-fold canonical UMAP split.
Objective: minimize OOF MAE (not RAE — per project memo
project_oof_lb_gap_is_rae_denominator.md: MAE transfers more cleanly
between OOF and LB).

Usage:
    pixi run python track1_activity/scripts/tune_mordred_boltz2_optuna.py --trials 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import psycopg2
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.joinpath("src")))

from data import get_conn, load_train_mordred, load_train_smiles_target  # noqa: E402
from splits import umap_split_indices  # noqa: E402

DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")
REPORT_DIR = ROOT.joinpath("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TIER1_PARQUET = ROOT.parent.joinpath("data", "boltz2_confidence_features.parquet")

BOLTZ2_SCALAR_COLS = [
    "affinity_pred_value", "affinity_pred_value_1", "affinity_pred_value_2",
    "affinity_probability_binary", "affinity_probability_binary_1",
    "affinity_probability_binary_2",
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    "ligand_atom_count", "ligand_to_pocket_distance_a",
]


def load_train_compound_ids() -> list[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()
    return ids


def load_boltz2_tier0(ids: list[int]) -> tuple[np.ndarray, list[str]]:
    cols_sql = ", ".join(f"b.{c}" for c in BOLTZ2_SCALAR_COLS)
    query = f"""
    SELECT c.id AS compound_id, {cols_sql},
           (b.affinity_pred_value_1 - b.affinity_pred_value_2) AS ensemble_diff_affinity,
           (b.affinity_probability_binary_1 - b.affinity_probability_binary_2) AS ensemble_diff_prob
    FROM compounds c
    LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
    WHERE c.id = ANY(%s) ORDER BY c.id
    """
    with psycopg2.connect(**DB_KW) as conn:
        df = pd.read_sql(query, conn, params=(ids,))
    df = df.set_index("compound_id").reindex(ids)
    names = list(BOLTZ2_SCALAR_COLS) + ["ensemble_diff_affinity", "ensemble_diff_prob"]
    return df[names].to_numpy(dtype=np.float32), names


def lgbm_cv_mae(X, y, splits, feature_names, params, early_stop=100):
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, va in splits:
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=feature_names)
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr, feature_name=feature_names)
        bo = lgb.train(
            params, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va] = bo.predict(X[va], num_iteration=bo.best_iteration)
    return oof


def full_metrics(y, p):
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RAE": float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - np.mean(y)))),
        "Pearson": float(pearsonr(y, p)[0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("=== Loading data ===")
    train_df = load_train_smiles_target()
    train_ids = load_train_compound_ids()
    smiles = train_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)

    mordred_train, _ = load_train_mordred()
    mord_cols = list(mordred_train.columns)
    X_mord = np.nan_to_num(
        mordred_train.loc[train_ids, mord_cols].to_numpy(dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    X_t0, n_t0 = load_boltz2_tier0(train_ids)
    X_t1 = pd.read_parquet(TIER1_PARQUET).reindex(train_ids).to_numpy(dtype=np.float32)
    t1_names = list(pd.read_parquet(TIER1_PARQUET).columns)

    X = np.concatenate([X_mord, X_t0, X_t1], axis=1)
    names = mord_cols + n_t0 + t1_names
    print(f"Feature matrix: {X.shape}")

    print(f"\n=== Building UMAP split (seed={args.seed}) ===")
    splits = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=args.seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": args.seed,
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.12, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 60
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
        }
        oof = lgbm_cv_mae(X, y, splits, names, params)
        mae = float(np.mean(np.abs(y - oof)))
        trial.set_user_attr("RAE", float(
            np.sum(np.abs(y - oof)) / np.sum(np.abs(y - np.mean(y)))
        ))
        trial.set_user_attr("Pearson", float(pearsonr(y, oof)[0]))
        return mae

    # Baseline (default params) for reference
    default_params = dict(
        objective="regression", metric="mae", boosting_type="gbdt",
        verbose=-1, seed=args.seed, num_leaves=63, learning_rate=0.05,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        min_child_samples=20,
    )
    print("\n=== Baseline (default params) ===")
    t0 = time.time()
    base_oof = lgbm_cv_mae(X, y, splits, names, default_params)
    print(f"  baseline CV: {time.time() - t0:.1f}s")
    base = full_metrics(y, base_oof)
    print(f"  baseline MAE={base['MAE']:.4f}  RAE={base['RAE']:.4f}  Pearson={base['Pearson']:.4f}")

    print(f"\n=== Optuna ({args.trials} trials) ===")
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"\nOptuna done in {elapsed / 60:.1f} min")

    print("\n=== Best trial ===")
    print(f"  MAE = {study.best_value:.4f}")
    print(f"  RAE = {study.best_trial.user_attrs['RAE']:.4f}")
    print(f"  Pearson = {study.best_trial.user_attrs['Pearson']:.4f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")

    # Full metrics for best params — re-run to get OOF array
    best_params = dict(default_params, **study.best_params)
    print("\n=== Refit with best params (full metrics) ===")
    best_oof = lgbm_cv_mae(X, y, splits, names, best_params)
    best = full_metrics(y, best_oof)
    print(f"  MAE={best['MAE']:.4f}  RAE={best['RAE']:.4f}  Pearson={best['Pearson']:.4f}")

    improvement = {
        "delta_MAE": base["MAE"] - best["MAE"],
        "delta_RAE": base["RAE"] - best["RAE"],
        "delta_Pearson": best["Pearson"] - base["Pearson"],
    }
    print(f"  Improvement: MAE {improvement['delta_MAE']:+.4f}, "
          f"RAE {improvement['delta_RAE']:+.4f}, "
          f"Pearson {improvement['delta_Pearson']:+.4f}")

    # Save
    trial_rows = [
        {
            "number": t.number,
            "value_MAE": t.value,
            "RAE": t.user_attrs.get("RAE"),
            "Pearson": t.user_attrs.get("Pearson"),
            **t.params,
        }
        for t in study.trials
    ]
    pd.DataFrame(trial_rows).to_csv(
        REPORT_DIR.joinpath("tune_mordred_boltz2_optuna_trials.csv"), index=False
    )
    pd.DataFrame([
        {"model": "baseline", **base},
        {"model": "best", **best},
    ]).to_csv(
        REPORT_DIR.joinpath("tune_mordred_boltz2_optuna_summary.csv"), index=False
    )
    np.save(REPORT_DIR.joinpath("tune_mordred_boltz2_optuna_oof.npy"), best_oof)

    import json
    with open(REPORT_DIR.joinpath("tune_mordred_boltz2_optuna_best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved to {REPORT_DIR}/tune_mordred_boltz2_optuna_*")


if __name__ == "__main__":
    main()
