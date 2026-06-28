#!/usr/bin/env python
"""Evaluate composite scalar gates with Phase 2 fold-wise calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop/pool_composite_scores.csv"
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
    / "composite_pairrank_chemprop_oof"
)

BASE_SCORE_COLS = [
    "pairrank_chembl",
    "pairrank_htchem",
    "pairrank_all",
    "cp_abs005",
    "cp_abs01",
    "cp_abs02",
]
COMPOSITE_COLS = [
    "pairrank_chembl",
    "pairrank_htchem",
    "cp_abs01",
    "cp_abs02",
    "combo_high_chembl_cp01",
    "combo_high_chembl_cp02",
    "combo_high_htchem_cp02",
    "combo_low_chembl_cp01",
    "combo_low_chembl_cp005",
]
FIXED_GATES = [
    {
        "name": "as1_combo_high_htchem_cp02_q97_lift030",
        "score": "combo_high_htchem_cp02",
        "mode": "high_lift",
        "q": 0.97,
        "shift": 0.30,
    },
    {
        "name": "as1_pairrank_chembl_q95_lift030",
        "score": "pairrank_chembl",
        "mode": "high_lift",
        "q": 0.95,
        "shift": 0.30,
    },
    {
        "name": "as1_cp_abs01_q95_drop030",
        "score": "cp_abs01",
        "mode": "low_drop",
        "q": 0.95,
        "shift": -0.30,
    },
    {
        "name": "oof_pairrank_htchem_q85_drop010",
        "score": "pairrank_htchem",
        "mode": "low_drop",
        "q": 0.85,
        "shift": -0.10,
    },
]
TRUE_BINS = ["lt3", "3to4", "4to5", "5to6", "gte6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(stats.spearmanr(y, pred).statistic) if len(y) > 1 else np.nan,
    }


def tail_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    high = (y >= 6.0).astype(int)
    low = (y < 3.0).astype(int)
    return {
        "spearman": float(stats.spearmanr(score, y).statistic),
        "gte6_auc": float(roc_auc_score(high, score))
        if high.min() != high.max()
        else np.nan,
        "gte6_ap": float(average_precision_score(high, score))
        if high.min() != high.max()
        else np.nan,
        "lt3_auc": float(roc_auc_score(low, -score))
        if low.min() != low.max()
        else np.nan,
        "lt3_ap": float(average_precision_score(low, -score))
        if low.min() != low.max()
        else np.nan,
    }


def add_foldwise_scores(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    for col in COMPOSITE_COLS:
        out[f"oof_{col}"] = np.nan

    for fold in sorted(out["fold"].unique()):
        train = out[~out["fold"].eq(fold)]
        val_mask = out["fold"].eq(fold)
        means = train[BASE_SCORE_COLS].mean()
        stds = train[BASE_SCORE_COLS].std(ddof=0).replace(0.0, np.nan)
        z = (out.loc[val_mask, BASE_SCORE_COLS] - means) / stds
        for col in BASE_SCORE_COLS:
            out.loc[val_mask, f"oof_{col}"] = z[col]
        out.loc[val_mask, "oof_combo_high_chembl_cp01"] = (
            z["pairrank_chembl"] + z["cp_abs01"]
        )
        out.loc[val_mask, "oof_combo_high_chembl_cp02"] = (
            z["pairrank_chembl"] + z["cp_abs02"]
        )
        out.loc[val_mask, "oof_combo_high_htchem_cp02"] = (
            z["pairrank_htchem"] + z["cp_abs02"]
        )
        out.loc[val_mask, "oof_combo_low_chembl_cp01"] = (
            -z["pairrank_chembl"] - z["cp_abs01"]
        )
        out.loc[val_mask, "oof_combo_low_chembl_cp005"] = (
            -z["pairrank_chembl"] - z["cp_abs005"]
        )

    missing = [
        col for col in out.columns if col.startswith("oof_") and out[col].isna().any()
    ]
    if missing:
        raise RuntimeError(f"Missing fold-wise scores: {missing}")
    return out


def summarize_scores(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = pool["pec50"].to_numpy(dtype=float)
    for col in COMPOSITE_COLS:
        rows.append(
            {"score": col, **tail_metrics(y, pool[f"oof_{col}"].to_numpy(dtype=float))}
        )
    return pd.DataFrame(rows).sort_values("gte6_ap", ascending=False)


def scan_foldwise_gates(pool: pd.DataFrame) -> pd.DataFrame:
    y = pool["pec50"].to_numpy(dtype=float)
    base = pool["phase2_oof_pred"].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(base - y)))
    rows = []
    for score_col in COMPOSITE_COLS:
        for mode, sign in [("high_lift", 1.0), ("low_drop", -1.0)]:
            for q in [0.85, 0.90, 0.93, 0.95, 0.97]:
                for shift in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
                    pred = base.copy()
                    flag = np.zeros(len(pool), dtype=bool)
                    for fold in sorted(pool["fold"].unique()):
                        train = pool[~pool["fold"].eq(fold)]
                        val_mask = pool["fold"].eq(fold).to_numpy()
                        train_score = train[f"oof_{score_col}"].to_numpy(dtype=float)
                        val_score = pool.loc[val_mask, f"oof_{score_col}"].to_numpy(
                            dtype=float
                        )
                        train_oriented = (
                            train_score if mode == "high_lift" else -train_score
                        )
                        val_oriented = val_score if mode == "high_lift" else -val_score
                        threshold = float(np.quantile(train_oriented, q))
                        local_flag = val_oriented >= threshold
                        flag[val_mask] = local_flag
                    pred[flag] += sign * shift
                    mae = float(np.mean(np.abs(pred - y)))
                    rows.append(
                        {
                            "score": score_col,
                            "mode": mode,
                            "q": q,
                            "shift": sign * shift,
                            "mae": mae,
                            "delta": mae - base_mae,
                            "n": int(flag.sum()),
                            "high": int(((y >= 6.0) & flag).sum()),
                            "low": int(((y < 3.0) & flag).sum()),
                            "as1_flags": int(
                                (pool["source"].eq("as1").to_numpy() & flag).sum()
                            ),
                            "as1_high": int(
                                (
                                    pool["source"].eq("as1").to_numpy()
                                    & (y >= 6.0)
                                    & flag
                                ).sum()
                            ),
                            "as1_low": int(
                                (
                                    pool["source"].eq("as1").to_numpy()
                                    & (y < 3.0)
                                    & flag
                                ).sum()
                            ),
                            "mean_base_error": float(np.mean(base[flag] - y[flag]))
                            if flag.any()
                            else np.nan,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["delta", "n"])


def summarize_oof(pool: pd.DataFrame, pred_col: str) -> pd.DataFrame:
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


def foldwise_flag(
    pool: pd.DataFrame, score_col: str, mode: str, q: float
) -> np.ndarray:
    flag = np.zeros(len(pool), dtype=bool)
    for fold in sorted(pool["fold"].unique()):
        train = pool[~pool["fold"].eq(fold)]
        val_mask = pool["fold"].eq(fold).to_numpy()
        train_score = train[f"oof_{score_col}"].to_numpy(dtype=float)
        val_score = pool.loc[val_mask, f"oof_{score_col}"].to_numpy(dtype=float)
        train_oriented = train_score if mode == "high_lift" else -train_score
        val_oriented = val_score if mode == "high_lift" else -val_score
        flag[val_mask] = val_oriented >= float(np.quantile(train_oriented, q))
    return flag


def evaluate_fixed_gates(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pool["pec50"].to_numpy(dtype=float)
    base = pool["phase2_oof_pred"].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(base - y)))
    rows = []
    slice_rows = []
    for spec in FIXED_GATES:
        flag = foldwise_flag(
            pool, str(spec["score"]), str(spec["mode"]), float(spec["q"])
        )
        pred = base.copy()
        pred[flag] += float(spec["shift"])
        mae = float(np.mean(np.abs(pred - y)))
        rows.append(
            {
                **spec,
                "mae": mae,
                "delta": mae - base_mae,
                "n": int(flag.sum()),
                "high": int(((y >= 6.0) & flag).sum()),
                "low": int(((y < 3.0) & flag).sum()),
                "as1_flags": int((pool["source"].eq("as1").to_numpy() & flag).sum()),
                "as1_high": int(
                    (pool["source"].eq("as1").to_numpy() & (y >= 6.0) & flag).sum()
                ),
                "as1_low": int(
                    (pool["source"].eq("as1").to_numpy() & (y < 3.0) & flag).sum()
                ),
                "mean_base_error": float(np.mean(base[flag] - y[flag]))
                if flag.any()
                else np.nan,
            }
        )
        tmp = pool.copy()
        tmp["fixed_gate_pred"] = pred
        summary = summarize_oof(tmp, "fixed_gate_pred")
        summary.insert(0, "name", str(spec["name"]))
        slice_rows.append(summary)
    return (
        pd.DataFrame(rows).sort_values("delta"),
        pd.concat(slice_rows, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool_scores = pd.read_csv(args.pool)
    folds = pd.read_csv(args.folds)[["pool_idx", "fold"]]
    oof = pd.read_csv(args.oof)[["pool_idx", "phase2_oof_pred"]]
    pool = pool_scores.merge(folds, on="pool_idx", how="inner").merge(
        oof, on="pool_idx", how="inner"
    )
    if len(pool) != len(pool_scores):
        raise RuntimeError(f"Aligned {len(pool)} of {len(pool_scores)} pool rows")

    pool = add_foldwise_scores(pool)
    score_summary = summarize_scores(pool)
    gate_scan = scan_foldwise_gates(pool)
    fixed_summary, fixed_slice_summary = evaluate_fixed_gates(pool)
    best = gate_scan.iloc[0]
    pred = pool["phase2_oof_pred"].to_numpy(dtype=float).copy()
    score_col = str(best["score"])
    mode = str(best["mode"])
    flag = foldwise_flag(pool, score_col, mode, float(best["q"]))
    pred[flag] += float(best["shift"])
    pool["best_gate_pred"] = pred
    pool["best_gate_flag"] = flag
    base_summary = summarize_oof(pool, "phase2_oof_pred").assign(model="base_oof")
    best_summary = summarize_oof(pool, "best_gate_pred").assign(model="best_gate")
    oof_summary = pd.concat([base_summary, best_summary], ignore_index=True)

    pool.to_csv(args.out_dir / "pool_oof_composite_scores.csv", index=False)
    score_summary.to_csv(args.out_dir / "oof_score_summary.csv", index=False)
    gate_scan.to_csv(args.out_dir / "oof_gate_scan.csv", index=False)
    fixed_summary.to_csv(args.out_dir / "fixed_gate_summary.csv", index=False)
    fixed_slice_summary.to_csv(
        args.out_dir / "fixed_gate_slice_summary.csv", index=False
    )
    oof_summary.to_csv(args.out_dir / "oof_summary.csv", index=False)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pool": str(args.pool),
                "folds": str(args.folds),
                "oof": str(args.oof),
                "best_gate": best.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(score_summary.to_string(index=False))
    print("\nBest gates")
    print(gate_scan.head(20).to_string(index=False))
    print("\nFixed gates")
    print(fixed_summary.to_string(index=False))
    print("\nOOF summary")
    print(oof_summary.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
