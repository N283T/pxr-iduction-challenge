#!/usr/bin/env python
"""TabPFN readouts on composite pairrank/ChemProp scalar signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_composite_residual_probe import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_OOF,
    DEFAULT_OUT_DIR,
    DEFAULT_POOL,
    FEATURE_COLS,
    make_features,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = DEFAULT_OUT_DIR.parent / "composite_pairrank_chemprop_tabpfn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cap", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_regressor(args: argparse.Namespace, seed: int):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model_path = TabPFNRegressor.create_default_for_version(ModelVersion.V3).model_path
    return TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=True,
        show_progress_bar=False,
    )


def make_classifier(args: argparse.Namespace, seed: int):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model_path = TabPFNClassifier.create_default_for_version(ModelVersion.V3).model_path
    return TabPFNClassifier(
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=True,
        show_progress_bar=False,
    )


def run_oof(
    pool: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pool["pec50"].to_numpy(dtype=float)
    base = pool["phase2_oof_pred"].to_numpy(dtype=float)
    residual = y - base
    classes = np.select([y < 3.0, y >= 6.0], [0, 2], default=1)
    reg_shift = np.full(len(pool), np.nan, dtype=float)
    clf_shift = np.full(len(pool), np.nan, dtype=float)
    rows = []
    for fold in sorted(pool["fold"].unique()):
        train_mask = ~pool["fold"].eq(fold)
        val_mask = pool["fold"].eq(fold)
        means = pool.loc[train_mask, FEATURE_COLS].mean()
        stds = pool.loc[train_mask, FEATURE_COLS].std(ddof=0).replace(0.0, np.nan)
        x_train = make_features(pool.loc[train_mask], means, stds)
        x_val = make_features(pool.loc[val_mask], means, stds)

        reg = make_regressor(args, args.seed + int(fold))
        reg.fit(x_train, residual[train_mask])
        reg_raw = reg.predict(x_val)
        reg_shift[val_mask] = np.clip(reg_raw, -args.cap, args.cap)

        clf = make_classifier(args, args.seed + int(fold))
        clf.fit(x_train, classes[train_mask])
        proba = clf.predict_proba(x_val)
        clf_shift[val_mask] = np.clip(
            args.cap * (proba[:, 2] - proba[:, 0]), -args.cap, args.cap
        )

        rows.append(
            {
                "fold": int(fold),
                "n_val": int(val_mask.sum()),
                "mae_base": float(np.mean(np.abs(base[val_mask] - y[val_mask]))),
                "mae_tabpfn_reg": float(
                    np.mean(np.abs(base[val_mask] + reg_shift[val_mask] - y[val_mask]))
                ),
                "mae_tabpfn_clf": float(
                    np.mean(np.abs(base[val_mask] + clf_shift[val_mask] - y[val_mask]))
                ),
                "reg_mean_abs_shift": float(np.mean(np.abs(reg_shift[val_mask]))),
                "clf_mean_abs_shift": float(np.mean(np.abs(clf_shift[val_mask]))),
            }
        )
    out = pool.copy()
    out["tabpfn_reg_shift"] = reg_shift
    out["tabpfn_reg_pred"] = base + reg_shift
    out["tabpfn_clf_shift"] = clf_shift
    out["tabpfn_clf_pred"] = base + clf_shift
    return out, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_scores = pd.read_csv(args.pool)
    folds = pd.read_csv(args.folds)[["pool_idx", "fold"]]
    oof = pd.read_csv(args.oof)[["pool_idx", "phase2_oof_pred"]]
    pool = pool_scores.merge(folds, on="pool_idx").merge(oof, on="pool_idx")
    out, fold_metrics = run_oof(pool, args)
    summary = pd.concat(
        [
            summarize(out, "phase2_oof_pred").assign(model="base"),
            summarize(out, "tabpfn_reg_pred").assign(model="tabpfn_residual"),
            summarize(out, "tabpfn_clf_pred").assign(model="tabpfn_tail_classifier"),
        ],
        ignore_index=True,
    )
    out.to_csv(args.out_dir / "pool_tabpfn_oof.csv", index=False)
    summary.to_csv(args.out_dir / "oof_summary.csv", index=False)
    fold_metrics.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pool": str(args.pool),
                "folds": str(args.folds),
                "oof": str(args.oof),
                "cap": args.cap,
                "device": args.device,
                "n_estimators": args.n_estimators,
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
