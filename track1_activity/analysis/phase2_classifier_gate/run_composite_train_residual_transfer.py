#!/usr/bin/env python
"""Train residual adapters on train rows only and transfer to AS1/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_composite_residual_probe import (  # noqa: E402
    DEFAULT_ANCHOR,
    DEFAULT_OOF,
    DEFAULT_OUT_DIR,
    DEFAULT_POOL,
    DEFAULT_TEST,
    FEATURE_COLS,
    load_anchor,
    make_features,
    summarize,
)

OUT_DIR = DEFAULT_OUT_DIR.parent / "composite_pairrank_chemprop_train_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cap", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
    }


def as1_bin_rows(as1: pd.DataFrame, pred_col: str, anchor_col: str) -> pd.DataFrame:
    rows = []
    for label, sub in as1.groupby("true_bin", observed=True):
        pred_metrics = metric_row(
            sub["pec50"].to_numpy(dtype=float),
            sub[pred_col].to_numpy(dtype=float),
        )
        anchor_metrics = metric_row(
            sub["pec50"].to_numpy(dtype=float),
            sub[anchor_col].to_numpy(dtype=float),
        )
        rows.append(
            {
                "true_bin": str(label),
                **pred_metrics,
                "delta_mae_vs_anchor": pred_metrics["mae"] - anchor_metrics["mae"],
            }
        )
    return pd.DataFrame(rows)


def train_residual_model(train: pd.DataFrame, cap: float, seed: int):
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
    return model, means, stds


def predict_shift(
    model,
    frame: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
    base_col: str,
    cap: float,
) -> np.ndarray:
    tmp = frame.copy()
    tmp["phase2_oof_pred"] = tmp[base_col].to_numpy(dtype=float)
    return np.clip(model.predict(make_features(tmp, means, stds)), -cap, cap)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(args.pool).merge(
        pd.read_csv(args.oof)[["pool_idx", "phase2_oof_pred"]], on="pool_idx"
    )
    test = pd.read_csv(args.test)
    anchor = load_anchor(args.anchor)
    test = test.merge(
        anchor[["molecule_name", "SMILES", "anchor_pred"]], on="molecule_name"
    )
    test["split"] = np.where(test["pec50"].notna(), "AS1", "AS2")
    test["true_bin"] = pd.cut(
        test["pec50"],
        [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf],
        labels=["lt3", "3to4", "4to5", "5to6", "gte6"],
    ).astype("object")

    train = pool[pool["source"].eq("train")].copy()
    model, means, stds = train_residual_model(train, args.cap, args.seed)
    pool["train_transfer_shift"] = predict_shift(
        model, pool, means, stds, "phase2_oof_pred", args.cap
    )
    pool["train_transfer_pred"] = pool["phase2_oof_pred"] + pool["train_transfer_shift"]

    test["train_transfer_shift"] = predict_shift(
        model, test, means, stds, "anchor_pred", args.cap
    )
    test["train_transfer_pred"] = test["anchor_pred"] + test["train_transfer_shift"]
    candidate = test.rename(
        columns={"molecule_name": "Molecule Name", "train_transfer_pred": "pEC50"}
    )[["SMILES", "Molecule Name", "pEC50"]]

    as1 = test[test["split"].eq("AS1")].copy()
    as1_summary = pd.DataFrame(
        [
            {
                "model": "anchor",
                **metric_row(
                    as1["pec50"].to_numpy(dtype=float),
                    as1["anchor_pred"].to_numpy(dtype=float),
                ),
            },
            {
                "model": "train_transfer_residual",
                **metric_row(
                    as1["pec50"].to_numpy(dtype=float),
                    as1["train_transfer_pred"].to_numpy(dtype=float),
                ),
            },
        ]
    )
    train_summary = pd.concat(
        [
            summarize(train, "phase2_oof_pred").assign(model="train_base_oof"),
            summarize(pool, "train_transfer_pred").assign(
                model="train_transfer_residual"
            ),
        ],
        ignore_index=True,
    )
    as1_bins = as1_bin_rows(as1, "train_transfer_pred", "anchor_pred")

    pool.to_csv(args.out_dir / "pool_train_transfer.csv", index=False)
    test.to_csv(args.out_dir / "test_train_transfer.csv", index=False)
    candidate.to_csv(args.out_dir / "test_train_transfer_candidate.csv", index=False)
    as1_summary.to_csv(args.out_dir / "as1_summary.csv", index=False)
    as1_bins.to_csv(args.out_dir / "as1_bin_summary.csv", index=False)
    train_summary.to_csv(args.out_dir / "train_oof_summary.csv", index=False)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pool": str(args.pool),
                "test": str(args.test),
                "oof": str(args.oof),
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
    print("AS1 summary")
    print(as1_summary.to_string(index=False))
    print("\nAS1 bins")
    print(as1_bins.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
