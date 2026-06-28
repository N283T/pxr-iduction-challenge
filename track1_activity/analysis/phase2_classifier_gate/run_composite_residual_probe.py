#!/usr/bin/env python
"""Fold-safe residual probes on composite pairrank/ChemProp scalar signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop/pool_composite_scores.csv"
)
DEFAULT_TEST = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop/test_composite_scores.csv"
)
DEFAULT_FOLDS = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_validation_matrix/outputs/"
    / "phase2_labeled_pool_with_folds.csv"
)
DEFAULT_OOF = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_validation_matrix/outputs/"
    / "phase2_lgbm_topk_oof_predictions.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop_residual"
)
DEFAULT_ANCHOR = (
    REPO_ROOT / "track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv"
)

FEATURE_COLS = [
    "pairrank_chembl",
    "pairrank_htchem",
    "pairrank_all",
    "pairrank_single_conc",
    "cp_abs005",
    "cp_abs01",
    "cp_abs02",
    "combo_high_chembl_cp01",
    "combo_high_chembl_cp02",
    "combo_high_htchem_cp02",
]
TRUE_BINS = ["lt3", "3to4", "4to5", "5to6", "gte6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cap", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
    }


def summarize(pool: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    masks = {
        "all": pd.Series(True, index=pool.index),
        "source_train": pool["source"].eq("train"),
        "source_as1": pool["source"].eq("as1"),
        "true_lt3": pool["pec50"] < 3.0,
        "true_gte6": pool["pec50"] >= 6.0,
    }
    for label in TRUE_BINS:
        masks[f"bin_{label}"] = pool["true_bin"].eq(label)
    for name, mask in masks.items():
        sub = pool.loc[mask]
        rows.append(
            {
                "slice": name,
                **metric_row(
                    sub["pec50"].to_numpy(dtype=float),
                    sub[pred_col].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def make_features(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> np.ndarray:
    x = (df[FEATURE_COLS] - means) / stds
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    extra = pd.DataFrame(
        {
            "base_pred": df["phase2_oof_pred"].to_numpy(dtype=float),
            "base_x_combo_high": df["phase2_oof_pred"].to_numpy(dtype=float)
            * x["combo_high_htchem_cp02"].to_numpy(dtype=float),
            "base_x_cp_abs01": df["phase2_oof_pred"].to_numpy(dtype=float)
            * x["cp_abs01"].to_numpy(dtype=float),
        },
        index=df.index,
    )
    return pd.concat([x, extra], axis=1).to_numpy(dtype=np.float32)


def run_residual_oof(
    pool: pd.DataFrame, cap: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pool["pec50"].to_numpy(dtype=float)
    base = pool["phase2_oof_pred"].to_numpy(dtype=float)
    residual = y - base
    pred_residual = np.full(len(pool), np.nan, dtype=float)
    fold_rows = []
    for fold in sorted(pool["fold"].unique()):
        train_mask = ~pool["fold"].eq(fold)
        val_mask = pool["fold"].eq(fold)
        means = pool.loc[train_mask, FEATURE_COLS].mean()
        stds = pool.loc[train_mask, FEATURE_COLS].std(ddof=0).replace(0.0, np.nan)
        x_train = make_features(pool.loc[train_mask], means, stds)
        x_val = make_features(pool.loc[val_mask], means, stds)
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=5.0,
            objective="regression_l1",
            random_state=seed + int(fold),
            verbose=-1,
        )
        model.fit(
            x_train,
            residual[train_mask],
            eval_set=[(x_val, residual[val_mask])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        raw = model.predict(x_val)
        pred_residual[val_mask] = np.clip(raw, -cap, cap)
        fold_pred = base[val_mask] + pred_residual[val_mask]
        fold_rows.append(
            {
                "fold": int(fold),
                "best_iteration": int(model.best_iteration_ or 600),
                "n_val": int(val_mask.sum()),
                "mae_base": float(np.mean(np.abs(base[val_mask] - y[val_mask]))),
                "mae_residual": float(np.mean(np.abs(fold_pred - y[val_mask]))),
                "mean_abs_shift": float(np.mean(np.abs(pred_residual[val_mask]))),
            }
        )
    out = pool.copy()
    out["residual_shift"] = pred_residual
    out["residual_pred"] = base + pred_residual
    return out, pd.DataFrame(fold_rows)


def run_gate_oof(
    pool: pd.DataFrame, cap: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pool["pec50"].to_numpy(dtype=float)
    base = pool["phase2_oof_pred"].to_numpy(dtype=float)
    classes = np.select([y < 3.0, y >= 6.0], [0, 2], default=1)
    shift = np.zeros(len(pool), dtype=float)
    fold_rows = []
    for fold in sorted(pool["fold"].unique()):
        train_mask = ~pool["fold"].eq(fold)
        val_mask = pool["fold"].eq(fold)
        means = pool.loc[train_mask, FEATURE_COLS].mean()
        stds = pool.loc[train_mask, FEATURE_COLS].std(ddof=0).replace(0.0, np.nan)
        x_train = make_features(pool.loc[train_mask], means, stds)
        x_val = make_features(pool.loc[val_mask], means, stds)
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=5.0,
            class_weight={0: 2.0, 1: 1.0, 2: 4.0},
            objective="multiclass",
            random_state=seed + int(fold),
            verbose=-1,
        )
        model.fit(x_train, classes[train_mask])
        proba = model.predict_proba(x_val)
        local_shift = cap * (proba[:, 2] - proba[:, 0])
        shift[val_mask] = np.clip(local_shift, -cap, cap)
        fold_pred = base[val_mask] + shift[val_mask]
        fold_rows.append(
            {
                "fold": int(fold),
                "n_val": int(val_mask.sum()),
                "mae_base": float(np.mean(np.abs(base[val_mask] - y[val_mask]))),
                "mae_gate": float(np.mean(np.abs(fold_pred - y[val_mask]))),
                "mean_abs_shift": float(np.mean(np.abs(shift[val_mask]))),
            }
        )
    out = pool.copy()
    out["gate_shift"] = shift
    out["gate_pred"] = base + shift
    return out, pd.DataFrame(fold_rows)


def load_anchor(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(
        columns={"Molecule Name": "molecule_name", "pEC50": "anchor_pred"}
    )


def train_final_residual_candidate(
    pool: pd.DataFrame,
    test: pd.DataFrame,
    anchor: pd.DataFrame,
    cap: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pool.copy()
    means = train[FEATURE_COLS].mean()
    stds = train[FEATURE_COLS].std(ddof=0).replace(0.0, np.nan)
    residual = train["pec50"].to_numpy(dtype=float) - train["phase2_oof_pred"].to_numpy(
        dtype=float
    )
    x_train = make_features(train, means, stds)
    model = lgb.LGBMRegressor(
        n_estimators=260,
        learning_rate=0.02,
        num_leaves=7,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=5.0,
        objective="regression_l1",
        random_state=seed,
        verbose=-1,
    )
    model.fit(x_train, residual)

    test_base = test.merge(anchor[["molecule_name", "anchor_pred"]], on="molecule_name")
    if len(test_base) != len(test):
        raise RuntimeError(
            f"Aligned {len(test_base)} of {len(test)} test rows to anchor"
        )
    test_base["phase2_oof_pred"] = test_base["anchor_pred"]
    x_test = make_features(test_base, means, stds)
    shift = np.clip(model.predict(x_test), -cap, cap)
    test_base["residual_shift"] = shift
    test_base["candidate_pred"] = test_base["anchor_pred"] + shift
    anchor_cols = ["molecule_name"]
    if "SMILES" in anchor.columns:
        anchor_cols.append("SMILES")
    anchor_cols.append("anchor_pred")
    candidate = anchor[anchor_cols].merge(
        test_base[["molecule_name", "candidate_pred"]], on="molecule_name"
    )
    candidate = candidate.rename(
        columns={"molecule_name": "Molecule Name", "candidate_pred": "pEC50"}
    )
    output_cols = (
        ["SMILES", "Molecule Name", "pEC50"]
        if "SMILES" in candidate
        else ["Molecule Name", "pEC50"]
    )
    candidate = candidate[output_cols]
    return candidate, test_base


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_scores = pd.read_csv(args.pool)
    test_scores = pd.read_csv(args.test)
    anchor = load_anchor(args.anchor)
    folds = pd.read_csv(args.folds)[["pool_idx", "fold"]]
    oof = pd.read_csv(args.oof)[["pool_idx", "phase2_oof_pred"]]
    pool = pool_scores.merge(folds, on="pool_idx").merge(oof, on="pool_idx")
    if len(pool) != len(pool_scores):
        raise RuntimeError(f"Aligned {len(pool)} of {len(pool_scores)} rows")

    residual_oof, residual_folds = run_residual_oof(pool, args.cap, args.seed)
    gate_oof, gate_folds = run_gate_oof(pool, args.cap, args.seed)
    merged = residual_oof.merge(
        gate_oof[["pool_idx", "gate_shift", "gate_pred"]], on="pool_idx"
    )
    summary = pd.concat(
        [
            summarize(pool, "phase2_oof_pred").assign(model="base"),
            summarize(residual_oof, "residual_pred").assign(model="residual_lgbm"),
            summarize(gate_oof, "gate_pred").assign(model="tail_gate_lgbm"),
        ],
        ignore_index=True,
    )
    merged.to_csv(args.out_dir / "pool_residual_gate_oof.csv", index=False)
    summary.to_csv(args.out_dir / "oof_summary.csv", index=False)
    residual_folds.to_csv(args.out_dir / "residual_fold_metrics.csv", index=False)
    gate_folds.to_csv(args.out_dir / "gate_fold_metrics.csv", index=False)
    candidate, test_shift = train_final_residual_candidate(
        pool, test_scores, anchor, args.cap, args.seed
    )
    candidate.to_csv(args.out_dir / "test_residual_candidate.csv", index=False)
    test_shift.to_csv(args.out_dir / "test_residual_shift.csv", index=False)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pool": str(args.pool),
                "folds": str(args.folds),
                "oof": str(args.oof),
                "test": str(args.test),
                "anchor": str(args.anchor),
                "cap": args.cap,
                "seed": args.seed,
                "features": FEATURE_COLS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
