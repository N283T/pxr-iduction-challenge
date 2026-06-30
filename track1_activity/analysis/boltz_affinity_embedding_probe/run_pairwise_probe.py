#!/usr/bin/env -S pixi run python
"""Pairwise AS1 replay for Boltz affinity-module embeddings.

This is an experiment-only probe inspired by Boltz-2's affinity training
objective. It fits pair-difference models on original train_activity rows and
evaluates all AS1 compound pairs.

No submission files or experiment DB rows are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs_pairwise"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

import run_probe  # noqa: E402
from data import load_train_smiles_target  # noqa: E402


def build_y_by_compound() -> dict[int, float]:
    train_df = load_train_smiles_target()
    train_ids = run_probe.load_ids("train_activity")
    return dict(
        zip(train_ids, train_df["pec50"].to_numpy(dtype=np.float32), strict=True)
    )


def sample_train_pairs(
    y: np.ndarray,
    n_pairs: int,
    seed: int,
    min_abs_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left_parts = []
    right_parts = []
    total = 0
    batch = max(n_pairs * 2, 100_000)
    n = len(y)
    while total < n_pairs:
        i = rng.integers(0, n, size=batch, endpoint=False)
        j = rng.integers(0, n, size=batch, endpoint=False)
        mask = i != j
        if min_abs_delta > 0:
            mask &= np.abs(y[i] - y[j]) >= min_abs_delta
        i = i[mask]
        j = j[mask]
        need = n_pairs - total
        if len(i) > need:
            i = i[:need]
            j = j[:need]
        left_parts.append(i.astype(np.int32))
        right_parts.append(j.astype(np.int32))
        total += len(i)
    return np.concatenate(left_parts), np.concatenate(right_parts)


def as1_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def pair_matrix(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return (X[i] - X[j]).astype(np.float32, copy=False)


def pair_metrics(
    y_delta: np.ndarray,
    cls_prob: np.ndarray,
    reg_delta: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    mask = np.abs(y_delta) >= threshold
    yt = y_delta[mask]
    cp = cls_prob[mask]
    rd = reg_delta[mask]
    y_cls = yt > 0
    cls_pred = cp >= 0.5
    reg_sign = rd > 0
    out: dict[str, float | int] = {
        "abs_delta_threshold": float(threshold),
        "n_pairs": int(mask.sum()),
        "class_accuracy": float(accuracy_score(y_cls, cls_pred)),
        "reg_sign_accuracy": float(accuracy_score(y_cls, reg_sign)),
        "delta_mae": float(np.mean(np.abs(rd - yt))),
        "delta_bias": float(np.mean(rd - yt)),
    }
    if len(np.unique(y_cls)) == 2:
        out["class_auc"] = float(roc_auc_score(y_cls, cp))
    else:
        out["class_auc"] = float("nan")
    if len(yt) >= 2 and np.std(rd) > 0:
        out["delta_pearson"] = float(stats.pearsonr(yt, rd).statistic)
        out["delta_spearman"] = float(stats.spearmanr(yt, rd).statistic)
    else:
        out["delta_pearson"] = float("nan")
        out["delta_spearman"] = float("nan")
    return out


def train_pair_models(
    X_pair: np.ndarray,
    y_delta: np.ndarray,
    seed: int,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor, dict[str, float]]:
    y_cls = y_delta > 0
    tr_idx, va_idx = train_test_split(
        np.arange(len(y_delta)),
        test_size=0.2,
        random_state=seed,
        stratify=y_cls,
    )
    clf = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.01,
        reg_lambda=1.0,
        objective="binary",
        random_state=seed,
        verbose=-1,
    )
    clf.fit(
        X_pair[tr_idx],
        y_cls[tr_idx],
        eval_set=[(X_pair[va_idx], y_cls[va_idx])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    reg = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.01,
        reg_lambda=1.0,
        objective="regression_l1",
        random_state=seed,
        verbose=-1,
    )
    reg.fit(
        X_pair[tr_idx],
        y_delta[tr_idx],
        eval_set=[(X_pair[va_idx], y_delta[va_idx])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_prob = clf.predict_proba(X_pair[va_idx])[:, 1]
    val_delta = reg.predict(X_pair[va_idx])
    val_metrics = {
        "val_class_accuracy": float(accuracy_score(y_cls[va_idx], val_prob >= 0.5)),
        "val_class_auc": float(roc_auc_score(y_cls[va_idx], val_prob)),
        "val_reg_sign_accuracy": float(accuracy_score(y_cls[va_idx], val_delta > 0)),
        "val_delta_mae": float(np.mean(np.abs(val_delta - y_delta[va_idx]))),
        "clf_best_iteration": int(clf.best_iteration_ or clf.n_estimators),
        "reg_best_iteration": int(reg.best_iteration_ or reg.n_estimators),
    }
    return clf, reg, val_metrics


def run_feature(
    feature: run_probe.FeatureConfig,
    args: argparse.Namespace,
    as1: pd.DataFrame,
    y_by_compound: dict[int, float],
    out_dir: Path,
) -> pd.DataFrame:
    print(f"\n=== {feature.name} ===")
    X_train, X_as1, train_ids = run_probe.load_feature(feature, as1)
    X_train, X_as1 = run_probe.finite_impute(X_train, X_as1)
    y_train = np.asarray(
        [y_by_compound[int(cid)] for cid in train_ids], dtype=np.float32
    )
    y_as1 = as1["pec50"].to_numpy(dtype=np.float32)
    print(f"X_train={X_train.shape} X_as1={X_as1.shape}")

    pair_i, pair_j = sample_train_pairs(
        y_train,
        n_pairs=args.n_pairs,
        seed=args.seed,
        min_abs_delta=args.min_abs_delta,
    )
    y_delta_train = (y_train[pair_i] - y_train[pair_j]).astype(np.float32)
    X_pair_train = pair_matrix(X_train, pair_i, pair_j)
    print(
        f"train pairs={len(y_delta_train)} "
        f"mean_abs_delta={float(np.mean(np.abs(y_delta_train))):.3f}"
    )

    clf, reg, val_metrics = train_pair_models(X_pair_train, y_delta_train, args.seed)
    del X_pair_train

    as1_i, as1_j = as1_pair_indices(len(as1))
    X_pair_as1 = pair_matrix(X_as1, as1_i, as1_j)
    y_delta_as1 = (y_as1[as1_i] - y_as1[as1_j]).astype(np.float32)
    cls_prob = clf.predict_proba(X_pair_as1)[:, 1]
    reg_delta = reg.predict(X_pair_as1)

    pred_df = pd.DataFrame(
        {
            "left_compound_id": as1.iloc[as1_i]["compound_id"].to_numpy(),
            "right_compound_id": as1.iloc[as1_j]["compound_id"].to_numpy(),
            "left_pec50": y_as1[as1_i],
            "right_pec50": y_as1[as1_j],
            "true_delta": y_delta_as1,
            "class_prob_left_gt_right": cls_prob,
            "pred_delta": reg_delta,
        }
    )
    pred_path = out_dir / f"{feature.name}__pair_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    rows = []
    for threshold in args.eval_thresholds:
        rows.append(
            {
                "feature": feature.name,
                "n_features": int(X_train.shape[1]),
                "train_rows": int(len(X_train)),
                "train_pairs": int(args.n_pairs),
                "train_min_abs_delta": float(args.min_abs_delta),
                **val_metrics,
                **pair_metrics(y_delta_as1, cls_prob, reg_delta, threshold),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{feature.name}__pair_summary.csv", index=False)
    print(
        summary[summary["abs_delta_threshold"].eq(0.5)][
            [
                "feature",
                "n_pairs",
                "class_accuracy",
                "class_auc",
                "reg_sign_accuracy",
                "delta_mae",
                "delta_spearman",
            ]
        ].to_string(index=False)
    )
    return summary


def write_report(out_dir: Path) -> None:
    summary = pd.read_csv(out_dir / "summary.csv")
    key = summary[summary["abs_delta_threshold"].eq(0.5)].sort_values(
        ["class_auc", "delta_mae"], ascending=[False, True]
    )
    wide = summary.pivot_table(
        index=["feature", "n_features"],
        columns="abs_delta_threshold",
        values=["class_accuracy", "class_auc", "reg_sign_accuracy", "delta_mae"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}@{thr:g}" for metric, thr in wide.columns]
    wide = wide.reset_index()
    lines = [
        "# Boltz Affinity Embedding Pairwise AS1 Probe",
        "",
        "Pair models fit on sampled train_activity pairs; AS1 is held out and evaluated over all AS1 pairs.",
        "",
        "## Main Readout",
        "",
        key[
            [
                "feature",
                "n_features",
                "n_pairs",
                "class_accuracy",
                "class_auc",
                "reg_sign_accuracy",
                "delta_mae",
                "delta_spearman",
                "val_class_auc",
                "val_delta_mae",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Threshold Sweep",
        "",
        wide.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", nargs="*", default=["all"])
    parser.add_argument("--n-pairs", type=int, default=150_000)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument(
        "--eval-thresholds", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_DIR / (
        f"lgbm_pairs{args.n_pairs}_mindelta"
        f"{str(args.min_abs_delta).replace('.', 'p')}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    as1 = run_probe.load_as1()
    y_by_compound = build_y_by_compound()
    selected = run_probe.choose_features(args.feature)

    summaries = []
    for feature in selected:
        summary_path = out_dir / f"{feature.name}__pair_summary.csv"
        if summary_path.exists() and not args.force:
            print(f"skip existing {feature.name}")
            summaries.append(pd.read_csv(summary_path))
            continue
        summaries.append(run_feature(feature, args, as1, y_by_compound, out_dir))
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(out_dir / "summary.csv", index=False)
    metadata = {
        "n_pairs": args.n_pairs,
        "min_abs_delta": args.min_abs_delta,
        "eval_thresholds": args.eval_thresholds,
        "seed": args.seed,
        "features": [asdict(f) for f in selected],
        "note": "Experiment only; no submission files or experiment DB rows written.",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_report(out_dir)
    print(f"\nwrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
