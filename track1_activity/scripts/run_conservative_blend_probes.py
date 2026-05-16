"""Conservative blend diagnostics over the current Track 1 ensemble pool.

This script uses only existing member OOF predictions and submission CSVs. It
does not record DB experiments by default. The goal is to test low-risk blend
families that are anchored to the current `ens_caruana_bag20` weights rather
than searching for a new molecular axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "scripts"))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics, load_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

OUT_DIR = REPO_ROOT / "track1_activity" / "analysis" / "conservative_blends"
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"


@dataclass
class ProbeResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    weights: np.ndarray | None
    extra: dict[str, float | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--write-submissions",
        action="store_true",
        help="Write CSVs for selected conservative candidates.",
    )
    parser.add_argument(
        "--submission-prefix",
        default="ens_conservative_probe",
        help="Filename prefix used with --write-submissions.",
    )
    return parser.parse_args()


def load_latest_caruana_weight_map() -> dict[str, float]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT hyperparameters
              FROM experiments
             WHERE name = 'ens_caruana_bag20'
             ORDER BY id DESC
             LIMIT 1
            """
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("ens_caruana_bag20 not found in experiments")
    hp = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return {str(k): float(v) for k, v in hp["weights"].items()}


def load_pool_by_names(
    names: list[str], y_train: np.ndarray, n_test: int
) -> tuple[np.ndarray, np.ndarray]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, submission_path FROM experiments WHERE name = ANY(%s)",
            (names,),
        )
        rows = cur.fetchall()
    rows_by_name = {
        str(name): (int(exp_id), sub_path) for exp_id, name, sub_path in rows
    }
    missing = [name for name in names if name not in rows_by_name]
    if missing:
        raise RuntimeError(f"Missing experiments for current weights: {missing}")

    oofs = []
    tests = []
    for name in names:
        exp_id, sub_path = rows_by_name[name]
        oof = load_oof_predictions(exp_id)
        if oof is None or len(oof) != len(y_train):
            raise RuntimeError(
                f"{name}: invalid OOF length "
                f"{None if oof is None else len(oof)} != {len(y_train)}"
            )
        csv_path = REPO_ROOT / str(sub_path)
        df = pd.read_csv(csv_path)
        if len(df) != n_test or "pEC50" not in df.columns:
            raise RuntimeError(f"{name}: invalid submission CSV {csv_path}")
        oofs.append(oof.astype(np.float64))
        tests.append(df["pEC50"].to_numpy(dtype=np.float64))
    return np.column_stack(oofs), np.column_stack(tests)


def normalize_weight_map(weights_map: dict[str, float], names: list[str]) -> np.ndarray:
    weights = np.array([weights_map[name] for name in names], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        raise RuntimeError("Stored caruana weights sum to zero")
    return weights / total


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def fit_simplex_mae(
    X: np.ndarray,
    y: np.ndarray,
    anchor: np.ndarray,
    *,
    l2_anchor: float,
    max_weight: float | None = None,
) -> np.ndarray:
    n = X.shape[1]
    bounds = [(0.0, 1.0 if max_weight is None else max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def objective(w: np.ndarray) -> float:
        return mae(y, X @ w) + l2_anchor * float(np.sum((w - anchor) ** 2))

    result = minimize(
        objective,
        anchor.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def outer_weight_cv(
    name: str,
    X: np.ndarray,
    X_test: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    anchor: np.ndarray,
    *,
    l2_anchor: float,
    max_weight: float | None = None,
) -> ProbeResult:
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_weights = []
    for tr_idx, va_idx in splits:
        w = fit_simplex_mae(
            X[tr_idx],
            y[tr_idx],
            anchor,
            l2_anchor=l2_anchor,
            max_weight=max_weight,
        )
        oof[va_idx] = X[va_idx] @ w
        fold_weights.append(w)
    full_w = fit_simplex_mae(
        X,
        y,
        anchor,
        l2_anchor=l2_anchor,
        max_weight=max_weight,
    )
    test = X_test @ full_w
    fold_w = np.vstack(fold_weights)
    return ProbeResult(
        name=name,
        oof=oof,
        test=test,
        weights=full_w,
        extra={
            "l2_anchor": float(l2_anchor),
            "max_weight": "none" if max_weight is None else float(max_weight),
            "mean_weight_l1_from_anchor": float(np.abs(full_w - anchor).sum()),
            "fold_weight_l1_std": float(np.std(np.abs(fold_w - anchor).sum(axis=1))),
        },
    )


def fit_predict_meta(
    model_name: str,
    X: np.ndarray,
    X_test: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> ProbeResult:
    oof = np.full(len(y), np.nan, dtype=np.float64)

    for tr_idx, va_idx in splits:
        model = make_meta_model(model_name)
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X[va_idx])
    full_model = make_meta_model(model_name)
    full_model.fit(X, y)
    return ProbeResult(
        name=f"stack_{model_name}",
        oof=oof,
        test=full_model.predict(X_test),
        weights=None,
        extra={},
    )


def make_meta_model(model_name: str):
    if model_name == "ridge":
        return RidgeCV(alphas=np.logspace(-3, 3, 25))
    if model_name == "positive_linear":
        return LinearRegression(positive=True)
    if model_name == "huber":
        return make_pipeline(
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000),
        )
    raise ValueError(model_name)


def percentile_columns(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    out = np.zeros_like(query, dtype=np.float64)
    n_train = train.shape[0]
    for j in range(train.shape[1]):
        order = np.sort(train[:, j])
        out[:, j] = np.searchsorted(order, query[:, j], side="right") / n_train
    return out


def rank_average_probe(
    name: str,
    X: np.ndarray,
    X_test: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    *,
    calibrator: str,
) -> ProbeResult:
    q_all = percentile_columns(X, X)
    q_test = percentile_columns(X, X_test)
    score_all = q_all @ weights
    score_test = q_test @ weights
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for tr_idx, va_idx in splits:
        if calibrator == "linear":
            model = LinearRegression()
            model.fit(score_all[tr_idx].reshape(-1, 1), y[tr_idx])
            oof[va_idx] = model.predict(score_all[va_idx].reshape(-1, 1))
        elif calibrator == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(score_all[tr_idx], y[tr_idx])
            oof[va_idx] = model.predict(score_all[va_idx])
        else:
            raise ValueError(calibrator)

    if calibrator == "linear":
        full_model = LinearRegression()
        full_model.fit(score_all.reshape(-1, 1), y)
        test = full_model.predict(score_test.reshape(-1, 1))
    else:
        full_model = IsotonicRegression(out_of_bounds="clip")
        full_model.fit(score_all, y)
        test = full_model.predict(score_test)
    return ProbeResult(
        name=f"{name}_{calibrator}",
        oof=oof,
        test=test,
        weights=None,
        extra={"calibrator": calibrator},
    )


def movement_summary(anchor_test: np.ndarray, test: np.ndarray) -> dict[str, float]:
    delta = test - anchor_test
    return {
        "test_delta_mean": float(delta.mean()),
        "test_delta_std": float(delta.std()),
        "test_abs_delta_mean": float(np.mean(np.abs(delta))),
        "test_abs_delta_p95": float(np.quantile(np.abs(delta), 0.95)),
        "test_abs_delta_max": float(np.max(np.abs(delta))),
        "test_pred_mean": float(test.mean()),
        "test_pred_std": float(test.std()),
    }


def result_row(
    result: ProbeResult,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> dict[str, float | str]:
    metrics = compute_metrics(y, result.oof)
    delta_oof = result.oof - anchor_oof
    row: dict[str, float | str] = {
        "name": result.name,
        "MAE": float(metrics["MAE"]),
        "RAE": float(metrics["RAE"]),
        "R2": float(metrics["R2"]),
        "Spearman_R": float(metrics["Spearman_R"]),
        "delta_mae_vs_anchor": float(metrics["MAE"] - mae(y, anchor_oof)),
        "oof_abs_delta_mean": float(np.mean(np.abs(delta_oof))),
        "oof_abs_delta_p95": float(np.quantile(np.abs(delta_oof), 0.95)),
    }
    row.update(movement_summary(anchor_test, result.test))
    row.update(result.extra)
    return row


def print_top_weights(names: list[str], weights: np.ndarray, label: str) -> None:
    pairs = sorted(zip(names, weights), key=lambda item: -item[1])
    print(f"\n{label} weights:")
    for name, weight in pairs:
        if weight >= 0.01:
            print(f"  {name:<70} {weight:.4f}")


def write_submission(result: ProbeResult, test_df: pd.DataFrame, prefix: str) -> Path:
    out = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": result.test,
        }
    )
    path = SUBMISSION_DIR / f"{prefix}_{result.name}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    weights_map = load_latest_caruana_weight_map()
    names = list(weights_map.keys())
    X, X_test = load_pool_by_names(names, y, n_test=len(test_df))
    anchor_w = normalize_weight_map(weights_map, names)
    anchor_oof = X @ anchor_w
    anchor_test = X_test @ anchor_w
    splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    print(f"Loaded pool: {len(names)} members, X={X.shape}, test={X_test.shape}")
    print(f"Anchor MAE={mae(y, anchor_oof):.6f}")
    print_top_weights(names, anchor_w, "current ens_caruana_bag20")

    results: list[ProbeResult] = [
        ProbeResult(
            name="anchor_current_caruana",
            oof=anchor_oof,
            test=anchor_test,
            weights=anchor_w,
            extra={"note": "stored ens_caruana_bag20 weights"},
        )
    ]

    for l2 in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        results.append(
            outer_weight_cv(
                f"simplex_mae_anchor_l2_{str(l2).replace('.', 'p')}",
                X,
                X_test,
                y,
                splits,
                anchor_w,
                l2_anchor=l2,
            )
        )
    for cap in (0.25, 0.30, 0.35):
        results.append(
            outer_weight_cv(
                f"simplex_mae_anchor_l2_0p03_cap_{str(cap).replace('.', 'p')}",
                X,
                X_test,
                y,
                splits,
                anchor_w,
                l2_anchor=0.03,
                max_weight=cap,
            )
        )

    for model_name in ("ridge", "positive_linear", "huber"):
        results.append(fit_predict_meta(model_name, X, X_test, y, splits))

    uniform_w = np.ones(len(names), dtype=np.float64) / len(names)
    for weights_name, weights in (
        ("rank_anchor", anchor_w),
        ("rank_uniform", uniform_w),
    ):
        for calibrator in ("linear", "isotonic"):
            results.append(
                rank_average_probe(
                    weights_name,
                    X,
                    X_test,
                    y,
                    splits,
                    weights,
                    calibrator=calibrator,
                )
            )

    rows = [result_row(result, y, anchor_oof, anchor_test) for result in results]
    summary = pd.DataFrame(rows).sort_values(["MAE", "test_abs_delta_p95"])
    out_csv = args.out_dir / "conservative_blend_probe_summary.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== Summary ===")
    cols = [
        "name",
        "MAE",
        "delta_mae_vs_anchor",
        "Spearman_R",
        "test_abs_delta_mean",
        "test_abs_delta_p95",
        "test_abs_delta_max",
        "test_pred_std",
    ]
    print(summary[cols].to_markdown(index=False, floatfmt=".6f"))
    print(f"\nWrote {out_csv}")

    best = summary.iloc[0]["name"]
    for result in results:
        if result.name == best and result.weights is not None:
            print_top_weights(names, result.weights, f"best {best}")

    if args.write_submissions:
        safe = summary[
            (summary["delta_mae_vs_anchor"] <= -0.0005)
            & (summary["test_abs_delta_p95"] <= 0.12)
        ]["name"].tolist()
        print(f"\nWriting {len(safe)} conservative submissions")
        for result in results:
            if result.name in safe:
                path = write_submission(result, test_df, args.submission_prefix)
                print(f"  {result.name}: {path}")


if __name__ == "__main__":
    main()
