"""Evaluation and experiment tracking for Track 1."""

import json

import numpy as np
import psycopg2
from scipy import stats

from data import DB_PARAMS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute competition metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
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


def print_fold_summary(fold_metrics: list[dict]):
    """Print mean ± std across folds."""
    print("\n  Mean ± Std across folds:")
    for name in ["RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau"]:
        values = [m[name] for m in fold_metrics]
        print(f"    {name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def record_experiment(
    name: str,
    description: str,
    model_type: str,
    feature_set: str,
    hyperparameters: dict,
    fold_metrics: list[dict],
    submission_path: str | None = None,
    notes: str = "",
    num_boost_rounds: list[int] | None = None,
):
    """Record experiment and CV results to DB."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute(
        """INSERT INTO experiments (name, description, model_type, feature_set,
           hyperparameters, cv_folds, submission_path, notes)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING id""",
        (
            name,
            description,
            model_type,
            feature_set,
            json.dumps(hyperparameters),
            len(fold_metrics),
            submission_path,
            notes,
        ),
    )
    exp_id = cur.fetchone()[0]

    for i, m in enumerate(fold_metrics):
        nbr = num_boost_rounds[i] if num_boost_rounds else None
        cur.execute(
            """INSERT INTO experiment_cv_results
               (experiment_id, fold, mae, rae, r2, spearman_r, kendall_tau, num_boost_rounds)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                exp_id,
                i,
                float(m["MAE"]),
                float(m["RAE"]),
                float(m["R2"]),
                float(m["Spearman_R"]),
                float(m["Kendall_Tau"]),
                nbr,
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Recorded experiment '{name}' (id={exp_id})")
    return exp_id


def save_oof_predictions(
    experiment_id: int,
    oof_preds: np.ndarray,
    covered_mask: np.ndarray | None = None,
):
    """Save OOF predictions to DB for ensemble weight optimization.

    ``covered_mask`` is an optional boolean array of the same length as
    ``oof_preds``; when provided, only indices with ``covered_mask[i] ==
    True`` are persisted. This is required for partial-coverage CV
    splits (e.g. analog-aware) where most train rows are never in val
    and their ``oof_preds[i]`` value is a placeholder (zero), not a
    real prediction. Saving those zeros would corrupt downstream
    ensemble weight optimisation.
    """
    if covered_mask is not None and len(covered_mask) != len(oof_preds):
        raise ValueError(
            f"covered_mask length {len(covered_mask)} does not match "
            f"oof_preds length {len(oof_preds)}"
        )
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    for i, pred in enumerate(oof_preds):
        if covered_mask is not None and not covered_mask[i]:
            continue
        cur.execute(
            """INSERT INTO experiment_oof_predictions (experiment_id, train_idx, oof_prediction)
               VALUES (%s, %s, %s)
               ON CONFLICT (experiment_id, train_idx) DO UPDATE SET oof_prediction = EXCLUDED.oof_prediction""",
            (experiment_id, i, float(pred)),
        )
    conn.commit()
    cur.close()
    conn.close()


def load_oof_predictions(experiment_id: int) -> np.ndarray | None:
    """Load OOF predictions from DB."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        """SELECT oof_prediction FROM experiment_oof_predictions
           WHERE experiment_id = %s ORDER BY train_idx""",
        (experiment_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return None
    return np.array([r[0] for r in rows])
