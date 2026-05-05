#!/usr/bin/env -S pixi run python
"""Cross-fit residual heads on re-pooled Boltz trunk features.

This is a diagnostic/candidate generator for the 13k trunk-only feature axis.
It trains on residuals from an anchor ensemble OOF prediction, then searches a
small scalar correction on OOF before writing a corrected submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    load_oof_predictions,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402


FEATURE_PATH = REPO_ROOT.joinpath(
    "data", "boltz_affhead", "repooled_trunk_region_zstats.parquet"
)
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
REPORT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "boltz_trunk_residual_head", "outputs"
)


@dataclass(frozen=True)
class Anchor:
    exp_id: int
    name: str
    submission_path: Path
    oof: np.ndarray
    test_pred: np.ndarray


def latest_experiment(name: str) -> tuple[int, Path]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, submission_path
                FROM experiments
                WHERE name = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (name,),
            )
            row = cur.fetchone()
    if row is None:
        raise SystemExit(f"Anchor experiment not found: {name}")
    exp_id, rel_path = row
    return int(exp_id), REPO_ROOT.joinpath(rel_path)


def reconstruct_ensemble_oof(exp_id: int, n_train: int) -> np.ndarray | None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT model_type, hyperparameters FROM experiments WHERE id = %s",
                (exp_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            model_type, hyperparameters = row
    if model_type != "ensemble":
        return None
    params = hyperparameters
    if isinstance(params, str):
        params = json.loads(params)
    weights = params.get("weights", {})
    if not weights:
        return None

    blended = np.zeros(n_train, dtype=np.float64)
    for name, weight in weights.items():
        child_id, _ = latest_experiment(name)
        child_oof = load_oof_predictions(child_id)
        if child_oof is None:
            raise RuntimeError(
                f"Child ensemble member has no OOF: {name} id={child_id}"
            )
        if len(child_oof) != n_train:
            raise RuntimeError(
                f"Child OOF length mismatch: {name} has {len(child_oof)} != {n_train}"
            )
        blended += float(weight) * child_oof.astype(np.float64)
    return blended


def load_anchor(name: str, n_train: int) -> Anchor:
    exp_id, path = latest_experiment(name)
    oof = load_oof_predictions(exp_id)
    if oof is None:
        oof = reconstruct_ensemble_oof(exp_id, n_train)
    if oof is None:
        raise SystemExit(f"Anchor has no OOF predictions: {name} id={exp_id}")
    sub = pd.read_csv(path)
    return Anchor(
        exp_id=exp_id,
        name=name,
        submission_path=path,
        oof=oof.astype(np.float64),
        test_pred=sub["pEC50"].to_numpy(dtype=np.float64),
    )


def load_compound_ids() -> tuple[np.ndarray, np.ndarray]:
    train_ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", get_engine()
    )["compound_id"].to_numpy(dtype=np.int64)
    test_ids = pd.read_sql(
        "SELECT compound_id FROM test_activity ORDER BY id", get_engine()
    )["compound_id"].to_numpy(dtype=np.int64)
    return train_ids, test_ids


def load_feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    train_ids, test_ids = load_compound_ids()
    df = pd.read_parquet(FEATURE_PATH).set_index("compound_id")
    feature_cols = [c for c in df.columns if c != "recycling_steps"]

    def matrix(ids: np.ndarray) -> np.ndarray:
        x = df.reindex(ids)[feature_cols].to_numpy(dtype=np.float32).copy()
        col_mean = np.nanmean(x, axis=0)
        row, col = np.where(np.isnan(x))
        if len(row):
            x[row, col] = col_mean[col]
        return x

    return matrix(train_ids), matrix(test_ids)


def fit_predict_ridge(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(512, x_tr.shape[0] - 1, x_tr.shape[1])
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=42),
        RidgeCV(alphas=np.logspace(-3, 4, 16)),
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_va), model.predict(x_test)


def fit_predict_lgbm(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    params = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 40,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 8,
    }
    model = lgb.train(
        params,
        lgb.Dataset(x_tr, label=y_tr),
        num_boost_round=400,
    )
    return model.predict(x_va), model.predict(x_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default="ens_caruana_bag20")
    p.add_argument("--model", choices=["ridge", "lgbm"], default="ridge")
    p.add_argument("--name-suffix", default="")
    p.add_argument("--on-conflict-replace", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    anchor = load_anchor(args.anchor, len(y))
    if len(anchor.oof) != len(y):
        raise SystemExit(
            f"Anchor OOF length {len(anchor.oof)} != train length {len(y)}"
        )

    x_train, x_test = load_feature_matrix()
    residual = y - anchor.oof
    splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    resid_oof = np.zeros_like(y, dtype=np.float64)
    resid_test_folds = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if args.model == "ridge":
            va_pred, te_pred = fit_predict_ridge(
                x_train[tr_idx], residual[tr_idx], x_train[va_idx], x_test
            )
        else:
            va_pred, te_pred = fit_predict_lgbm(
                x_train[tr_idx],
                residual[tr_idx],
                x_train[va_idx],
                x_test,
                seed=42 + fold,
            )
        resid_oof[va_idx] = va_pred
        resid_test_folds.append(te_pred)
        print(f"fold {fold}: residual std pred={np.std(va_pred):.4f}")

    resid_test = np.mean(np.vstack(resid_test_folds), axis=0)
    base_metrics = compute_metrics(y, anchor.oof)
    resid_r = stats.pearsonr(residual, resid_oof).statistic
    print("\nAnchor OOF:")
    print_metrics(base_metrics)
    print(f"Residual OOF corr: {resid_r:.4f}")
    print(
        f"Residual pred std: train={np.std(resid_oof):.4f}, "
        f"test={np.std(resid_test):.4f}"
    )

    rows = []
    for alpha in np.linspace(-0.50, 0.50, 41):
        pred = anchor.oof + alpha * resid_oof
        m = compute_metrics(y, pred)
        rows.append(
            {
                "alpha": float(alpha),
                "mae": float(m["MAE"]),
                "rae": float(m["RAE"]),
                "spearman": float(m["Spearman_R"]),
                "delta_mae": float(m["MAE"] - base_metrics["MAE"]),
                "delta_spearman": float(m["Spearman_R"] - base_metrics["Spearman_R"]),
            }
        )
    grid = pd.DataFrame(rows).sort_values(["mae", "alpha"])
    best = grid.iloc[0]
    alpha = float(best["alpha"])
    corrected_oof = anchor.oof + alpha * resid_oof
    corrected_test = anchor.test_pred + alpha * resid_test

    suffix = args.name_suffix or f"{args.model}_a{alpha:+.2f}".replace(
        "+", "p"
    ).replace("-", "m")
    exp_name = f"ens_{anchor.name}_repooled_trunk_resid_{suffix}"
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": corrected_test,
        }
    ).to_csv(sub_path, index=False)

    corrected_metrics = compute_metrics(y, corrected_oof)
    print("\nBest corrected OOF:")
    print(best.to_string())
    print_metrics(corrected_metrics)
    print(f"Saved: {sub_path}")

    exp_id = record_experiment(
        name=exp_name,
        description=f"{args.model} residual head on re-pooled Boltz trunk features",
        model_type=f"residual_{args.model}",
        feature_set="repooled_trunk_region_zstats",
        hyperparameters={
            "anchor": anchor.name,
            "anchor_id": anchor.exp_id,
            "alpha": alpha,
            "residual_oof_corr": float(resid_r),
        },
        fold_metrics=[corrected_metrics],
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=(
            f"Anchor OOF MAE={base_metrics['MAE']:.4f}; corrected OOF "
            f"MAE={corrected_metrics['MAE']:.4f}; alpha={alpha:.2f}; "
            f"residual_oof_corr={resid_r:.4f}"
        ),
        on_conflict_replace=args.on_conflict_replace,
    )
    save_oof_predictions(exp_id, corrected_oof)

    report_path = REPORT_DIR.joinpath(f"{exp_name}.md")
    grid.to_csv(REPORT_DIR.joinpath(f"{exp_name}_alpha_grid.csv"), index=False)
    report_path.write_text(
        "\n".join(
            [
                f"# {exp_name}",
                "",
                f"- anchor: `{anchor.name}` id={anchor.exp_id}",
                f"- model: `{args.model}`",
                f"- residual_oof_corr: {resid_r:.4f}",
                f"- anchor_oof_mae: {base_metrics['MAE']:.4f}",
                f"- corrected_oof_mae: {corrected_metrics['MAE']:.4f}",
                f"- alpha: {alpha:.2f}",
                f"- submission: `{sub_path.relative_to(REPO_ROOT)}`",
                "",
                "## Best Alpha Grid Rows",
                "",
                grid.head(10).to_markdown(index=False),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
