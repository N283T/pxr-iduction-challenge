#!/usr/bin/env -S pixi run python
"""Ensemble v8: regularized weight optimization with explicit exclusion list.

Forks run_ensemble_v2.py (issue #20: OOF-LB gap from ensemble weight overfit)
and adds an exclusion list so known-bad experiments cannot accidentally
re-enter the ensemble:

- ``singleconc`` direct-feature variants (PR #41 — leaky, LB RAE 1.027)
- ``_pseudo`` label variants (PR #39 — documented negative)
- Default-params multitask runs without ``_tuned`` suffix (individually weak)

Strategies actually evaluated below:
1. Simple average over all models
2. Vanilla weight optimization (Nelder-Mead, no regularization)
3. L2-regularized weight optimization (penalize deviation from equal weights)
4. Fold-based weight optimization (optimize on each CV fold's val, avg weights)
5. Fold-based + L2
6. Top-K simple averages (k = 3, 5, 8)
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


# ---------------------------------------------------------------------------
# Data loading (reused from run_ensemble.py)
# ---------------------------------------------------------------------------


# Exclusion list: experiment name substrings that must NOT enter the ensemble.
# Update this list (and the module docstring) whenever a new failure mode is
# documented in docs/. Each excluded category corresponds to a published PR
# with the failure analysis.
EXCLUDE_SUBSTRINGS = (
    "singleconc",  # leaky direct-concat feature (PR #41, LB RAE 1.03)
    "_pseudo",  # pseudo-labeling variants (PR #39, documented negative)
)

# Conditional exclusions that don't fit a plain substring match. Each entry
# is a callable taking the experiment name and returning True to exclude.
EXCLUDE_PATTERNS = (
    # Default-params multitask runs: name starts with chemprop_multitask but
    # lacks the "_tuned" suffix added by run_chemprop_multitask.py --use-tuned.
    lambda n: n.startswith("chemprop_multitask") and "_tuned" not in n,
)


def _is_excluded(name: str) -> bool:
    if any(sub in name for sub in EXCLUDE_SUBSTRINGS):
        return True
    if any(pat(name) for pat in EXCLUDE_PATTERNS):
        return True
    return False


def find_experiments_with_oof() -> list[dict]:
    """Find all experiments with OOF predictions, excluding known-bad ones."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.id, e.name, e.feature_set, avg(cv.rae) AS avg_rae
        FROM experiments e
        JOIN experiment_cv_results cv ON cv.experiment_id = e.id
        WHERE EXISTS (SELECT 1 FROM experiment_oof_predictions oof WHERE oof.experiment_id = e.id)
          AND e.model_type NOT LIKE 'ensemble%'
        GROUP BY e.id, e.name, e.feature_set
        ORDER BY avg_rae
    """)
    raw = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    excluded = []
    for r in raw:
        entry = {
            "id": r[0],
            "name": r[1],
            "feature_set": r[2],
            "avg_rae": float(r[3]),
        }
        if _is_excluded(entry["name"]):
            excluded.append(entry)
        else:
            results.append(entry)

    if excluded:
        print(f"Excluded {len(excluded)} experiments:")
        for e in excluded:
            print(f"  [skip] {e['name']:<55} RAE={e['avg_rae']:.4f}")
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
        return None
    return pd.read_csv(sub_path)["pEC50"].values


def load_all_models(experiments, y_train):
    """Load OOF and test predictions for all valid models.

    Logs every drop with a reason so silently-missing models cannot
    sneak past the user.
    """
    model_names = []
    oof_list = []
    test_list = []
    dropped: list[tuple[str, str]] = []
    for exp in experiments:
        oof = load_oof_predictions(exp["id"])
        test_preds = load_submission_preds(exp["name"])
        if oof is None:
            dropped.append((exp["name"], "no OOF predictions in DB"))
            continue
        if test_preds is None:
            dropped.append((exp["name"], "no submission CSV on disk"))
            continue
        if len(oof) != len(y_train):
            dropped.append(
                (exp["name"], f"OOF length {len(oof)} != y_train {len(y_train)}")
            )
            continue
        model_names.append(exp["name"])
        oof_list.append(oof)
        test_list.append(test_preds)

    if dropped:
        print(f"\n[warn] Dropped {len(dropped)} models from ensemble:")
        for name, reason in dropped:
            print(f"  [drop] {name:<55} {reason}")
        if len(dropped) > 0.3 * len(experiments):
            print(
                f"[warn] Dropped >30% of candidate models "
                f"({len(dropped)}/{len(experiments)}). Check submissions/ "
                f"or DB consistency before trusting the result."
            )
    return model_names, np.column_stack(oof_list), np.column_stack(test_list)


# ---------------------------------------------------------------------------
# Weight optimization strategies
# ---------------------------------------------------------------------------


def normalize_weights(w):
    """L1-normalize the absolute weights so they sum to 1.

    Falls back to uniform weights if Nelder-Mead converges to all zeros
    (pathological but possible with high regularization). Prevents NaN
    propagation into ensemble predictions.
    """
    w_abs = np.abs(w)
    total = w_abs.sum()
    if total < 1e-12:
        return np.ones_like(w_abs) / len(w_abs)
    return w_abs / total


def optimize_weights_vanilla(oof_matrix, y_true):
    """Original: minimize RAE on full OOF (baseline, known to overfit)."""
    n = oof_matrix.shape[1]

    def objective(w):
        w_norm = normalize_weights(w)
        blended = oof_matrix @ w_norm
        return np.sum(np.abs(y_true - blended)) / np.sum(np.abs(y_true - y_true.mean()))

    result = minimize(
        objective,
        np.ones(n) / n,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return normalize_weights(result.x)


def optimize_weights_l2(oof_matrix, y_true, alpha=0.1):
    """L2-regularized: penalize deviation from equal weights."""
    n = oof_matrix.shape[1]
    equal_w = np.ones(n) / n

    def objective(w):
        w_norm = normalize_weights(w)
        blended = oof_matrix @ w_norm
        rae = np.sum(np.abs(y_true - blended)) / np.sum(np.abs(y_true - y_true.mean()))
        # L2 penalty toward equal weights
        penalty = alpha * np.sum((w_norm - equal_w) ** 2)
        return rae + penalty

    result = minimize(
        objective,
        equal_w.copy(),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return normalize_weights(result.x)


def optimize_weights_fold_based(oof_matrix, y_true, splits):
    """Optimize weights on each fold's validation set, then average weights."""
    n = oof_matrix.shape[1]
    all_weights = []

    for train_idx, val_idx in splits:
        oof_val = oof_matrix[val_idx]
        y_val = y_true[val_idx]

        def objective(w):
            w_norm = normalize_weights(w)
            blended = oof_val @ w_norm
            return np.sum(np.abs(y_val - blended)) / np.sum(
                np.abs(y_val - y_val.mean())
            )

        result = minimize(
            objective,
            np.ones(n) / n,
            method="Nelder-Mead",
            options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
        )
        all_weights.append(normalize_weights(result.x))

    # Average weights across folds
    avg_weights = np.mean(all_weights, axis=0)
    return avg_weights / avg_weights.sum()


def optimize_weights_fold_l2(oof_matrix, y_true, splits, alpha=0.1):
    """Fold-based + L2 regularization combined."""
    n = oof_matrix.shape[1]
    equal_w = np.ones(n) / n
    all_weights = []

    for train_idx, val_idx in splits:
        oof_val = oof_matrix[val_idx]
        y_val = y_true[val_idx]

        def objective(w):
            w_norm = normalize_weights(w)
            blended = oof_val @ w_norm
            rae = np.sum(np.abs(y_val - blended)) / np.sum(np.abs(y_val - y_val.mean()))
            penalty = alpha * np.sum((w_norm - equal_w) ** 2)
            return rae + penalty

        result = minimize(
            objective,
            equal_w.copy(),
            method="Nelder-Mead",
            options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
        )
        all_weights.append(normalize_weights(result.x))

    avg_weights = np.mean(all_weights, axis=0)
    return avg_weights / avg_weights.sum()


# ---------------------------------------------------------------------------
# Ensemble creation and evaluation
# ---------------------------------------------------------------------------


def evaluate_ensemble(
    name,
    weights,
    oof_matrix,
    y_true,
    test_matrix,
    test_df,
    model_names,
    record=True,
):
    """Evaluate and optionally record an ensemble."""
    blended_oof = oof_matrix @ weights
    metrics = compute_metrics(y_true, blended_oof)

    print(f"\n  {name}:")
    print(
        f"    OOF RAE={metrics['RAE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"R2={metrics['R2']:.4f}  Spearman={metrics['Spearman_R']:.4f}"
    )

    # Show non-trivial weights
    significant = [(n, w) for n, w in zip(model_names, weights) if w > 0.02]
    significant.sort(key=lambda x: -x[1])
    print(f"    Weights ({len(significant)} significant):")
    for n, w in significant:
        print(f"      {n:<35} {w:.4f}")

    # Generate submission
    blended_test = test_matrix @ weights
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": blended_test,
        }
    )
    sub_filename = f"{name}.csv"
    sub.to_csv(SUBMISSION_DIR.joinpath(sub_filename), index=False)

    if record:
        weight_dict = {n: float(w) for n, w in zip(model_names, weights)}
        record_experiment(
            name=name,
            description=f"Ensemble v8: {name}",
            model_type="ensemble_v8",
            feature_set="weighted_blend",
            hyperparameters={"weights": weight_dict, "strategy": name},
            fold_metrics=[metrics],
            submission_path=f"track1_activity/submissions/{sub_filename}",
            notes=f"OOF RAE={metrics['RAE']:.4f}",
        )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].values

    # CV splits for fold-based optimization
    scaffold_splits = scaffold_split_indices(
        train_df["smiles"].tolist(), n_splits=5, seed=42
    )

    # Load models
    experiments = find_experiments_with_oof()
    print(f"Found {len(experiments)} experiments with OOF predictions:")
    for exp in experiments:
        print(f"  {exp['name']:<35} RAE={exp['avg_rae']:.4f}")

    model_names, oof_matrix, test_matrix = load_all_models(experiments, y_train)
    print(f"\n{len(model_names)} models loaded.\n")

    n_models = len(model_names)
    results = {}

    # === 1. Simple average (all models) ===
    print("=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)

    equal_w = np.ones(n_models) / n_models
    results["simple_avg_all"] = evaluate_ensemble(
        "ens_v8_simple_avg",
        equal_w,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # === 2. Vanilla optimization (baseline, known to overfit) ===
    w_vanilla = optimize_weights_vanilla(oof_matrix, y_train)
    results["vanilla_opt"] = evaluate_ensemble(
        "ens_v8_vanilla_opt",
        w_vanilla,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # === 3. L2-regularized optimization ===
    for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
        w_l2 = optimize_weights_l2(oof_matrix, y_train, alpha=alpha)
        results[f"l2_alpha={alpha}"] = evaluate_ensemble(
            f"ens_v8_l2_a{alpha}",
            w_l2,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # === 4. Fold-based optimization ===
    w_fold = optimize_weights_fold_based(oof_matrix, y_train, scaffold_splits)
    results["fold_based"] = evaluate_ensemble(
        "ens_v8_fold_opt",
        w_fold,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # === 5. Fold-based + L2 ===
    for alpha in [0.1, 0.3]:
        w_fold_l2 = optimize_weights_fold_l2(
            oof_matrix, y_train, scaffold_splits, alpha=alpha
        )
        results[f"fold_l2_alpha={alpha}"] = evaluate_ensemble(
            f"ens_v8_fold_l2_a{alpha}",
            w_fold_l2,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # === 6. Top-K simple average ===
    for k in [3, 5, 8]:
        if k > n_models:
            continue
        topk_w = np.zeros(n_models)
        topk_w[:k] = 1.0 / k
        results[f"top{k}_avg"] = evaluate_ensemble(
            f"ens_v8_top{k}_avg",
            topk_w,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # === Summary ===
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<35} {'OOF RAE':>8} {'OOF MAE':>8} {'OOF R2':>8}")
    print(f"  {'-' * 60}")

    for name, m in sorted(results.items(), key=lambda x: x[1]["RAE"]):
        print(f"  {name:<35} {m['RAE']:8.4f} {m['MAE']:8.4f} {m['R2']:8.4f}")

    print(f"\n  LB RAE (reference): 0.62")
    print(f"\n  Recommendation: strategies with HIGHER OOF RAE may generalize")
    print(f"  better to LB if the current gap is caused by OOF overfitting.")
    print(f"  Submit the simple avg and the best regularized version to compare.")


if __name__ == "__main__":
    main()
