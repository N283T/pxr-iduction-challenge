"""Weighted ensemble of diverse single-feature models.

Loads OOF predictions from DB, optimizes weights, generates submission.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target
from evaluate import (
    compute_metrics,
    print_metrics,
    load_oof_predictions,
    record_experiment,
)
from splits import scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")


def find_single_feature_experiments() -> list[dict]:
    """Find all single-feature Optuna experiments with OOF predictions."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.id, e.name, e.feature_set, avg(cv.rae) AS avg_rae
        FROM experiments e
        JOIN experiment_cv_results cv ON cv.experiment_id = e.id
        WHERE EXISTS (SELECT 1 FROM experiment_oof_predictions oof WHERE oof.experiment_id = e.id)
        GROUP BY e.id, e.name, e.feature_set
        ORDER BY avg_rae
    """)
    results = [
        {"id": r[0], "name": r[1], "feature_set": r[2], "avg_rae": r[3]}
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return results


def load_submission_preds(experiment_name: str) -> np.ndarray | None:
    """Load test predictions from submission CSV."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT submission_path FROM experiments WHERE name = %s", (experiment_name,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None or row[0] is None:
        return None

    sub_path = Path(__file__).resolve().parent.parent.parent.joinpath(row[0])
    if not sub_path.exists():
        print(f"  WARNING: {sub_path} not found")
        return None
    return pd.read_csv(sub_path)["pEC50"].values


def optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Optimize ensemble weights by minimizing RAE on OOF predictions."""
    n_models = oof_matrix.shape[1]

    def objective(weights):
        w = np.abs(weights) / np.abs(weights).sum()
        blended = oof_matrix @ w
        return np.sum(np.abs(y_true - blended)) / np.sum(np.abs(y_true - y_true.mean()))

    result = minimize(
        objective,
        np.ones(n_models) / n_models,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    weights = np.abs(result.x) / np.abs(result.x).sum()
    return weights


def main():
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].values

    outer_splits = scaffold_split_indices(
        train_df["smiles"].tolist(), n_splits=5, seed=42
    )

    # Find models
    experiments = find_single_feature_experiments()
    if len(experiments) < 2:
        print(f"Only {len(experiments)} models with OOF predictions. Need at least 2.")
        return

    print(f"\nFound {len(experiments)} models with OOF predictions:")
    for exp in experiments:
        print(f"  {exp['name']:<35} RAE={exp['avg_rae']:.4f}  ({exp['feature_set']})")

    # Load OOF and test predictions
    model_names = []
    oof_list = []
    test_list = []

    for exp in experiments:
        oof = load_oof_predictions(exp["id"])
        test_preds = load_submission_preds(exp["name"])
        if oof is None or test_preds is None:
            print(f"  Skipping {exp['name']}: missing predictions")
            continue
        if len(oof) != len(y_train):
            print(
                f"  Skipping {exp['name']}: OOF length mismatch ({len(oof)} vs {len(y_train)})"
            )
            continue
        model_names.append(exp["name"])
        oof_list.append(oof)
        test_list.append(test_preds)

    print(f"\n{len(model_names)} models loaded successfully.")

    oof_matrix = np.column_stack(oof_list)
    test_matrix = np.column_stack(test_list)

    # --- 1. Simple average ---
    print(f"\n{'=' * 60}")
    print(f"  Simple Average ({len(model_names)} models)")
    print(f"{'=' * 60}")

    avg_oof = oof_matrix.mean(axis=1)
    avg_metrics = compute_metrics(y_train, avg_oof)
    print_metrics(avg_metrics, label="OOF")

    avg_test = test_matrix.mean(axis=1)
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": avg_test,
        }
    )
    sub.to_csv(SUBMISSION_DIR.joinpath("ensemble_avg_all.csv"), index=False)

    # --- 2. Optimized weights (all models) ---
    print(f"\n{'=' * 60}")
    print(f"  Optimized Weights ({len(model_names)} models)")
    print(f"{'=' * 60}")

    weights = optimize_weights(oof_matrix, y_train)
    weighted_oof = oof_matrix @ weights
    weighted_metrics = compute_metrics(y_train, weighted_oof)
    print_metrics(weighted_metrics, label="OOF")

    print("\n  Weights:")
    for name, w in sorted(zip(model_names, weights), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"    {name:<35} {w:.4f}")

    weighted_test = test_matrix @ weights
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": weighted_test,
        }
    )
    sub.to_csv(SUBMISSION_DIR.joinpath("ensemble_weighted_all.csv"), index=False)

    record_experiment(
        name="ensemble_weighted_all",
        description=f"Weighted ensemble of {len(model_names)} single-feature Optuna models",
        model_type="ensemble",
        feature_set="weighted_blend",
        hyperparameters={
            "weights": {n: float(w) for n, w in zip(model_names, weights)}
        },
        fold_metrics=[weighted_metrics],
        submission_path="track1_activity/submissions/ensemble_weighted_all.csv",
        notes=f"OOF RAE={weighted_metrics['RAE']:.4f}, {len(model_names)} models",
    )

    # --- 3. Top-K ensembles ---
    sorted_indices = np.argsort(
        [exp["avg_rae"] for exp in experiments if exp["name"] in model_names]
    )

    for k in [3, 5, 8, 10]:
        if k > len(model_names):
            continue

        top_idx = sorted_indices[:k]
        top_names = [model_names[i] for i in top_idx]

        # Simple average top-K
        topk_avg_oof = oof_matrix[:, top_idx].mean(axis=1)
        topk_avg_m = compute_metrics(y_train, topk_avg_oof)

        # Optimized top-K
        topk_weights = optimize_weights(oof_matrix[:, top_idx], y_train)
        topk_weighted_oof = oof_matrix[:, top_idx] @ topk_weights
        topk_weighted_m = compute_metrics(y_train, topk_weighted_oof)

        print(
            f"\n  Top-{k}: avg RAE={topk_avg_m['RAE']:.4f}, weighted RAE={topk_weighted_m['RAE']:.4f}"
        )

        topk_weighted_test = test_matrix[:, top_idx] @ topk_weights
        sub = pd.DataFrame(
            {
                "SMILES": test_df["smiles"],
                "Molecule Name": test_df["molecule_name"],
                "pEC50": topk_weighted_test,
            }
        )
        sub.to_csv(
            SUBMISSION_DIR.joinpath(f"ensemble_weighted_top{k}.csv"), index=False
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  ENSEMBLE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Simple avg (all):    RAE={avg_metrics['RAE']:.4f}")
    print(f"  Weighted (all):      RAE={weighted_metrics['RAE']:.4f}")
    print(
        f"  Best single model:   RAE={experiments[0]['avg_rae']:.4f} ({experiments[0]['name']})"
    )


if __name__ == "__main__":
    main()
