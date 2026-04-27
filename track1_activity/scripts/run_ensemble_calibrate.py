"""Post-hoc regression calibration of ens_caruana_bag20 output.

Fits Linear + Isotonic calibrators on the ensemble's 4140-point OOF
predictions vs. true pEC50. Evaluates each calibrator via 5-fold UMAP
nested CV (honest estimate, guards against calibration overfitting),
then refits on full OOF and applies to raw test predictions to produce
two calibrated submission CSVs.

Inputs:
- Per-member OOF predictions from experiment_oof_predictions (10 models)
- Caruana weights from the latest ens_caruana_bag20 row's hyperparameters
- True pEC50 from train_activity via data.load_train_smiles_target
- Raw test predictions: track1_activity/submissions/ens_caruana_bag20.csv

Outputs:
- track1_activity/submissions/ens_caruana_bag20_calibrated_linear.csv
- track1_activity/submissions/ens_caruana_bag20_calibrated_isotonic.csv
- Two experiments in DB: ens_caruana_bag20_calibrated_{linear,isotonic}

Usage:
    pixi run python track1_activity/scripts/run_ensemble_calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_train_smiles_target, load_test_smiles  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def load_latest_caruana_weights() -> dict[str, float]:
    """Fetch weights dict from the most recent ens_caruana_bag20 row."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT hyperparameters FROM experiments "
        "WHERE name = 'ens_caruana_bag20' ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        raise RuntimeError("No ens_caruana_bag20 row in DB. Run run_ensemble.py first.")
    hp = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return hp["weights"]


def load_member_oof_matrix(member_names: list[str], n_train: int) -> np.ndarray:
    """Build (n_train, n_members) OOF matrix, columns aligned to member_names.

    Each column is the OOF predictions for the named experiment, read from
    experiment_oof_predictions ORDER BY train_idx. Raises if any member has
    missing OOF rows.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cols: list[np.ndarray] = []
    for name in member_names:
        cur.execute(
            "SELECT oof_prediction FROM experiment_oof_predictions "
            "WHERE experiment_id = (SELECT id FROM experiments WHERE name = %s) "
            "ORDER BY train_idx",
            (name,),
        )
        rows = cur.fetchall()
        if len(rows) != n_train:
            cur.close()
            conn.close()
            raise RuntimeError(
                f"Member '{name}' has {len(rows)} OOF rows, expected {n_train}"
            )
        cols.append(np.array([r[0] for r in rows], dtype=np.float64))
    cur.close()
    conn.close()
    return np.stack(cols, axis=1)


def reconstruct_ensemble_oof(
    X_oof: np.ndarray, weights: dict[str, float], member_names: list[str]
) -> np.ndarray:
    """Compute y_pred_oof = sum(weights[name] * X_oof[:, i]) for i, name."""
    w = np.array([weights[n] for n in member_names], dtype=np.float64)
    return X_oof @ w


def load_raw_test_predictions() -> pd.DataFrame:
    """Read the latest ens_caruana_bag20.csv submission. Returns DataFrame with
    SMILES, Molecule Name, pEC50 columns (513 rows)."""
    path = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")
    if not path.exists():
        raise RuntimeError(f"Missing {path}. Run run_ensemble.py to regenerate.")
    df = pd.read_csv(path)
    if len(df) != 513:
        raise RuntimeError(f"Expected 513 test rows, got {len(df)}")
    return df


def _fit_spline_k5(y_pred: np.ndarray, y_true: np.ndarray, k: int = 5) -> dict:
    """Fit a k-knot monotone PCHIP spline on quantile bins.

    Uses OOF prediction quantiles as knot x-positions and bin-mean pEC50 as
    knot y-values. Monotonicity of y-values is enforced via cumulative max
    before PCHIP interpolation, so the resulting spline is monotone
    non-decreasing.
    """
    quantiles = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(y_pred, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(y_pred, edges) - 1, 0, k - 1)

    xs = np.zeros(k)
    ys = np.zeros(k)
    for kk in range(k):
        mask = bin_idx == kk
        if mask.sum() < 2:
            xs[kk] = 0.5 * (edges[kk] + edges[kk + 1])
            ys[kk] = ys[kk - 1] if kk > 0 else float(y_true.min())
        else:
            xs[kk] = float(y_pred[mask].mean())
            ys[kk] = float(y_true[mask].mean())

    # Enforce x strictly increasing (ties from tight quantile bins would
    # break PchipInterpolator).
    for i in range(1, k):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6

    # Enforce monotone non-decreasing y so the spline preserves order.
    ys = np.maximum.accumulate(ys)

    spline = PchipInterpolator(xs, ys, extrapolate=False)
    return {
        "xs": xs,
        "ys": ys,
        "spline": spline,
        "x_min": float(xs[0]),
        "x_max": float(xs[-1]),
    }


def fit_calibrator(method: str, y_pred: np.ndarray, y_true: np.ndarray):
    """Fit a calibrator and return an object with a .predict(x) method."""
    if method == "linear":
        model = LinearRegression()
        model.fit(y_pred.reshape(-1, 1), y_true)
        return model
    if method == "linear_pos":
        # Positive-constrained affine: slope >= 0 guarantees order preservation
        # on the OOF range. Intercept is unconstrained.
        X = np.column_stack([y_pred, np.ones_like(y_pred)])
        res = lsq_linear(X, y_true, bounds=([0.0, -np.inf], [np.inf, np.inf]))
        return {"slope": float(res.x[0]), "intercept": float(res.x[1])}
    if method == "spline_k5":
        return _fit_spline_k5(y_pred, y_true, k=5)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(y_pred, y_true)
        return model
    raise ValueError(f"Unknown method: {method}")


def apply_calibrator(method: str, model, y_pred: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to y_pred, returning the calibrated values."""
    if method == "linear":
        return model.predict(y_pred.reshape(-1, 1))
    if method == "linear_pos":
        return model["slope"] * y_pred + model["intercept"]
    if method == "spline_k5":
        y_clipped = np.clip(y_pred, model["x_min"], model["x_max"])
        return np.asarray(model["spline"](y_clipped), dtype=np.float64)
    if method == "isotonic":
        return model.predict(y_pred)
    raise ValueError(f"Unknown method: {method}")


def nested_cv_evaluate(
    y_pred_oof: np.ndarray,
    y_true: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
) -> tuple[list[dict], np.ndarray]:
    """5-fold nested CV: fit calibrator on 4 folds, evaluate on 1.

    Returns:
      - list of 5 metric dicts (one per fold)
      - calibrated OOF predictions (4140,) stitched from per-fold held-out predictions
    """
    calibrated_oof = np.zeros_like(y_pred_oof)
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        model = fit_calibrator(method, y_pred_oof[tr_idx], y_true[tr_idx])
        va_pred = apply_calibrator(method, model, y_pred_oof[va_idx])
        calibrated_oof[va_idx] = va_pred
        m = compute_metrics(y_true[va_idx], va_pred)
        fold_metrics.append(m)
        print_metrics(m, label=f"{method} Fold {fold}")
    return fold_metrics, calibrated_oof


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ensemble artifacts...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_true = train_df["pec50"].to_numpy(dtype=np.float64)
    n_train = len(y_true)
    print(f"  train rows: {n_train}, test rows: {len(test_df)}")

    weights = load_latest_caruana_weights()
    member_names = list(weights.keys())
    print(f"  caruana members: {len(member_names)}")
    for name in member_names:
        print(f"    {name:<50} w={weights[name]:.4f}")

    X_oof = load_member_oof_matrix(member_names, n_train)
    y_pred_oof = reconstruct_ensemble_oof(X_oof, weights, member_names)

    raw_oof_metrics = compute_metrics(y_true, y_pred_oof)
    print("\nReconstructed raw ensemble OOF:")
    print_metrics(raw_oof_metrics, label="RAW")

    # Sanity: raw OOF MAE should match the ens_caruana_bag20 DB value closely
    # (will differ very slightly due to float accumulation order).
    # Lower bound widened 2026-04-26 after Phase 4 Optuna deploy reached 0.392
    # (below the historical 0.40 floor).
    if not 0.36 <= raw_oof_metrics["MAE"] <= 0.45:
        raise RuntimeError(
            f"Raw OOF MAE {raw_oof_metrics['MAE']:.4f} outside expected "
            f"range [0.36, 0.45]. Weights/OOF misalignment?"
        )

    # Build 5-fold UMAP splits matching ENSEMBLE_MODELS training CV
    print("\nBuilding 5-fold UMAP splits (seed=42)...")
    splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    raw_test = load_raw_test_predictions()
    raw_test_pred = raw_test["pEC50"].to_numpy(dtype=np.float64)

    print("\n" + "=" * 60)
    print("  NESTED CV EVALUATION (5-fold UMAP)")
    print("=" * 60)

    method_results: dict[str, dict] = {}

    for method in ("linear", "linear_pos", "spline_k5", "isotonic"):
        print(f"\n--- {method} ---")
        fold_metrics, calibrated_oof = nested_cv_evaluate(
            y_pred_oof, y_true, splits, method
        )
        nested_cv_metrics = compute_metrics(y_true, calibrated_oof)
        print(f"\n  Overall nested-CV calibrated OOF ({method}):")
        print_metrics(nested_cv_metrics, label=f"{method} NESTED")

        # Sanity: Spearman should be essentially unchanged (monotonic calib)
        dspear = abs(nested_cv_metrics["Spearman_R"] - raw_oof_metrics["Spearman_R"])
        if dspear >= 0.005:
            print(
                f"  WARNING: Spearman changed by {dspear:.4f} "
                f"(expected < 0.005 for monotonic calibrator)"
            )

        # Refit on full OOF and apply to raw test predictions
        full_model = fit_calibrator(method, y_pred_oof, y_true)
        test_pred_cal = apply_calibrator(method, full_model, raw_test_pred)

        # Diagnostics on the fitted calibrator
        if method == "linear":
            fitted = {
                "slope": float(full_model.coef_[0]),
                "intercept": float(full_model.intercept_),
            }
            print(
                f"  Linear fit: slope={fitted['slope']:.4f}, "
                f"intercept={fitted['intercept']:.4f}"
            )
        elif method == "linear_pos":
            fitted = {
                "slope": full_model["slope"],
                "intercept": full_model["intercept"],
                "constraint_active": full_model["slope"] < 1e-12,
            }
            print(
                f"  Linear(pos) fit: slope={fitted['slope']:.4f}, "
                f"intercept={fitted['intercept']:.4f}, "
                f"constraint_active={fitted['constraint_active']}"
            )
        elif method == "spline_k5":
            fitted = {
                "n_knots": int(len(full_model["xs"])),
                "x_knots": full_model["xs"].tolist(),
                "y_knots": full_model["ys"].tolist(),
                "x_min": full_model["x_min"],
                "x_max": full_model["x_max"],
            }
            print(
                f"  Spline(k5) fit: n_knots={fitted['n_knots']}, "
                f"x_range=[{fitted['x_min']:.3f}, {fitted['x_max']:.3f}], "
                f"y_range=[{float(full_model['ys'].min()):.3f}, "
                f"{float(full_model['ys'].max()):.3f}]"
            )
        else:  # isotonic
            fitted = {
                "n_knots": int(len(full_model.X_thresholds_)),
                "x_min": float(full_model.X_min_),
                "x_max": float(full_model.X_max_),
                "y_min": float(full_model.y_thresholds_.min()),
                "y_max": float(full_model.y_thresholds_.max()),
            }
            print(
                f"  Isotonic fit: n_knots={fitted['n_knots']}, "
                f"x_range=[{fitted['x_min']:.3f}, {fitted['x_max']:.3f}], "
                f"y_range=[{fitted['y_min']:.3f}, {fitted['y_max']:.3f}]"
            )

        # Write submission CSV
        sub_name = f"ens_caruana_bag20_calibrated_{method}"
        sub_path = SUBMISSION_DIR.joinpath(f"{sub_name}.csv")
        sub_df = pd.DataFrame(
            {
                "SMILES": raw_test["SMILES"],
                "Molecule Name": raw_test["Molecule Name"],
                "pEC50": test_pred_cal,
            }
        )
        sub_df.to_csv(sub_path, index=False)
        print(f"  Wrote submission: {sub_path}")

        # Record experiment with nested-CV metrics as fold_metrics
        exp_id = record_experiment(
            name=sub_name,
            description=(
                f"Post-hoc {method} calibration of ens_caruana_bag20 "
                f"(5-fold UMAP nested CV)"
            ),
            model_type="ensemble_calibrated",
            feature_set="caruana_bag20_output",
            hyperparameters={
                "method": method,
                "fitted_params": fitted,
                "source_ensemble": "ens_caruana_bag20",
                "source_members": member_names,
                "source_weights": weights,
            },
            fold_metrics=fold_metrics,
            submission_path=f"track1_activity/submissions/{sub_name}.csv",
            notes=(
                f"Raw OOF MAE={raw_oof_metrics['MAE']:.4f}, "
                f"calibrated nested-CV MAE={nested_cv_metrics['MAE']:.4f}, "
                f"ΔMAE={nested_cv_metrics['MAE'] - raw_oof_metrics['MAE']:+.4f}, "
                f"{method} calibrator"
            ),
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, calibrated_oof)
        print(f"  Recorded experiment id={exp_id}")

        method_results[method] = {
            "nested_cv_metrics": nested_cv_metrics,
            "fold_metrics": fold_metrics,
            "calibrated_oof": calibrated_oof,
            "full_model": full_model,
            "fitted": fitted,
            "test_pred_cal": test_pred_cal,
        }

    # ---- Selection step: pick best by MAE with Spearman guardrail ---------
    print("\n" + "=" * 60)
    print("  SELECTION (MAE min, |ΔSpearman| < 0.005 guardrail)")
    print("=" * 60)
    raw_spearman = raw_oof_metrics["Spearman_R"]
    raw_mae = raw_oof_metrics["MAE"]

    summary_rows = []
    for name, res in method_results.items():
        m = res["nested_cv_metrics"]
        d_spear = abs(m["Spearman_R"] - raw_spearman)
        d_mae = m["MAE"] - raw_mae
        passes = (d_spear < 0.005) and (m["MAE"] < raw_mae)
        summary_rows.append((name, m["MAE"], d_mae, m["Spearman_R"], d_spear, passes))
        print(
            f"  {name:<12} MAE={m['MAE']:.4f} (Δ{d_mae:+.4f})  "
            f"Spearman={m['Spearman_R']:.4f} (Δ{d_spear:+.4f})  "
            f"passes_guardrail={passes}"
        )

    candidates = [r for r in summary_rows if r[5]]
    if not candidates:
        print(
            "\n  WARNING: No calibrator improves MAE with Spearman preserved. "
            "Skipping best submission CSV."
        )
        print("\nDone.")
        return

    best_name = min(candidates, key=lambda r: r[1])[0]
    best_res = method_results[best_name]
    print(f"\n  BEST: {best_name}")

    # Write single 'best' submission CSV
    best_sub_name = "ens_caruana_bag20_calibrated_best"
    best_sub_path = SUBMISSION_DIR.joinpath(f"{best_sub_name}.csv")
    best_sub_df = pd.DataFrame(
        {
            "SMILES": raw_test["SMILES"],
            "Molecule Name": raw_test["Molecule Name"],
            "pEC50": best_res["test_pred_cal"],
        }
    )
    best_sub_df.to_csv(best_sub_path, index=False)
    print(f"  Wrote best submission: {best_sub_path}")

    # Record meta experiment row pointing to the winning method
    best_exp_id = record_experiment(
        name=best_sub_name,
        description=(
            f"Best post-hoc calibration of ens_caruana_bag20 "
            f"(selected: {best_name}, 5-fold UMAP nested CV, MAE min + "
            f"|ΔSpearman| < 0.005 guardrail)"
        ),
        model_type="ensemble_calibrated",
        feature_set="caruana_bag20_output",
        hyperparameters={
            "selected_method": best_name,
            "fitted_params": best_res["fitted"],
            "candidates_considered": [r[0] for r in summary_rows],
            "selection_rule": "min MAE s.t. |Spearman - raw| < 0.005",
            "source_ensemble": "ens_caruana_bag20",
            "source_members": member_names,
            "source_weights": weights,
        },
        fold_metrics=best_res["fold_metrics"],
        submission_path=f"track1_activity/submissions/{best_sub_name}.csv",
        notes=(
            f"Raw OOF MAE={raw_mae:.4f}, "
            f"calibrated nested-CV MAE={best_res['nested_cv_metrics']['MAE']:.4f} "
            f"(method={best_name}), ΔMAE="
            f"{best_res['nested_cv_metrics']['MAE'] - raw_mae:+.4f}"
        ),
        on_conflict_replace=True,
    )
    save_oof_predictions(best_exp_id, best_res["calibrated_oof"])
    print(f"  Recorded best meta-experiment id={best_exp_id}")

    print("\nDone.")


if __name__ == "__main__":
    main()
