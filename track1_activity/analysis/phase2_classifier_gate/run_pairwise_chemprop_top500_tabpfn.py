#!/usr/bin/env -S pixi run python
"""Append pairwise ChemProp embeddings to the existing top500 TabPFN path.

This is a Phase 2 development diagnostic, not a submission generator. It uses
the train+AS1 labeled pool folds and mirrors the existing LGBM-gain top500
selection before fitting TabPFN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(
    0,
    str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_validation_matrix")),
)
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_classifier_gate"))
)
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from build_phase2_validation_matrix import (  # noqa: E402
    TRUE_BIN_LABELS,
    build_labeled_pool,
    build_phase2_feature_matrix,
    build_phase2_splits,
)
from run_pairwise_chemprop_embed_probe import extract_embeddings  # noqa: E402
from score_chemprop_pairwise_pretrain import load_pairwise_model  # noqa: E402

OUT_ROOT = (
    Path(__file__).resolve().parent / "outputs" / "pairwise_chemprop_top500_tabpfn"
)
DEFAULT_CKPT = REPO_ROOT.joinpath(
    "track1_activity",
    "checkpoints",
    "chemprop_pairwise_chembl_binding_random250k_100kp5",
    "pairwise_pretrain.pt",
)


def sanitize_matrix(x: np.ndarray) -> np.ndarray:
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    return np.where(np.isfinite(x), x, col_mean).astype(np.float32)


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    spearman = stats.spearmanr(y, pred).statistic if len(y) >= 2 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(spearman),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(y)),
    }


def summarize_oof(pool: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    y = pool["pec50"].to_numpy(dtype=np.float64)
    masks = {
        "all": np.ones(len(pool), dtype=bool),
        "source_train": pool["source"].eq("train").to_numpy(),
        "source_as1": pool["source"].eq("as1").to_numpy(),
        "true_lt3": y < 3.0,
        "true_gte6": y >= 6.0,
    }
    for label in TRUE_BIN_LABELS:
        masks[f"bin_{label}"] = pool["true_bin"].eq(label).to_numpy()
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        rows.append({"slice": name, **metric_row(y[mask], pred[mask])})
    return pd.DataFrame(rows)


def load_or_extract_pair_embeddings(
    pool: pd.DataFrame,
    ckpt: Path,
    cache_path: Path,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        emb = pd.read_parquet(cache_path)
        emb = emb.sort_values("pool_idx")
        if emb["pool_idx"].to_numpy().tolist() == pool["pool_idx"].to_numpy().tolist():
            cols = [c for c in emb.columns if c.startswith("pairchem_")]
            return emb[cols].to_numpy(dtype=np.float32)
    model = load_pairwise_model(ckpt)
    x_pair = extract_embeddings(model, pool["smiles"].astype(str).tolist(), batch_size)
    cols = [f"pairchem_{i:03d}" for i in range(x_pair.shape[1])]
    out = pd.DataFrame(x_pair, columns=cols)
    out.insert(0, "pool_idx", pool["pool_idx"].to_numpy())
    out.insert(1, "compound_id", pool["compound_id"].to_numpy())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return x_pair


def build_augmented_matrix(
    args: argparse.Namespace,
    pool: pd.DataFrame,
) -> tuple[np.ndarray, int, int]:
    x_base = build_phase2_feature_matrix(args.feature, pool)
    x_pair = load_or_extract_pair_embeddings(
        pool,
        args.ckpt,
        args.out_dir / "pool_pairchem_embeddings.parquet",
        args.batch_size,
        args.force_embeddings,
    )
    pair_start = x_base.shape[1]
    x = np.concatenate([x_base, x_pair], axis=1)
    return sanitize_matrix(x), pair_start, x_pair.shape[1]


def run_oof(args: argparse.Namespace) -> None:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = build_labeled_pool()
    splits, split_summary = build_phase2_splits(
        pool, n_splits=args.n_splits, n_clusters=args.n_clusters, seed=args.seed
    )
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)

    x, pair_start, n_pair = build_augmented_matrix(args, pool)
    y = pool["pec50"].to_numpy(dtype=np.float32)
    oof = np.full(len(pool), np.nan, dtype=np.float64)
    fold_rows = []
    selection_rows = []

    print(
        f"feature={args.feature} base+pairchem d={x.shape[1]} "
        f"pairchem_start={pair_start} n_pair={n_pair} top_k={args.top_k}"
    )
    for fold, (train_idx, val_idx) in enumerate(splits):
        ranker = lgb.LGBMRegressor(
            n_estimators=args.rank_n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=args.seed + fold,
            verbose=-1,
        )
        ranker.fit(x[train_idx], y[train_idx])
        gain = ranker.booster_.feature_importance(importance_type="gain")
        order = np.argsort(-gain)
        if args.pair_top_k > 0:
            base_order = order[order < pair_start]
            pair_order = order[(order >= pair_start) & (order < pair_start + n_pair)]
            selected = np.concatenate(
                [base_order[: args.top_k], pair_order[: args.pair_top_k]]
            ).astype(np.int64)
        else:
            selected = order[: args.top_k]
        pair_selected = selected[selected >= pair_start]
        total_gain = float(gain.sum())
        pair_gain = float(gain[pair_start : pair_start + n_pair].sum())
        selection_rows.append(
            {
                "fold": fold,
                "n_selected": int(len(selected)),
                "n_pairchem_selected": int(len(pair_selected)),
                "pairchem_gain_share_pct": float(pair_gain / total_gain * 100.0)
                if total_gain > 0
                else 0.0,
                "best_pairchem_rank": int(np.where(order >= pair_start)[0][0] + 1),
                "pairchem_selected_indices": ",".join(
                    str(int(i - pair_start)) for i in pair_selected[:50]
                ),
            }
        )
        model = TabPFNRegressor(
            device=args.device,
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
            average_before_softmax=args.average_before_softmax,
            random_state=args.seed + fold,
            model_path=model_path,
            ignore_pretraining_limits=args.top_k > 500,
        )
        model.fit(x[train_idx][:, selected], y[train_idx])
        pred = model.predict(x[val_idx][:, selected]).astype(np.float64)
        oof[val_idx] = pred
        fold_metric = metric_row(y[val_idx].astype(np.float64), pred)
        fold_rows.append({"fold": fold, **fold_metric})
        print(
            f"fold={fold} mae={fold_metric['mae']:.4f} "
            f"sp={fold_metric['spearman']:.4f} "
            f"pairchem_selected={len(pair_selected)} "
            f"pairchem_gain={selection_rows[-1]['pairchem_gain_share_pct']:.2f}%"
        )

    if np.isnan(oof).any():
        raise RuntimeError("incomplete OOF predictions")

    out_oof = pool.copy()
    out_oof["phase2_oof_pred"] = oof
    out_oof["phase2_oof_error"] = out_oof["phase2_oof_pred"] - out_oof["pec50"]
    out_oof["phase2_oof_abs_error"] = out_oof["phase2_oof_error"].abs()
    summary = summarize_oof(pool, oof)
    folds = pd.DataFrame(fold_rows)
    selection = pd.DataFrame(selection_rows)

    out_oof.to_csv(args.out_dir / "oof_predictions.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    folds.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    selection.to_csv(args.out_dir / "selection.csv", index=False)
    meta = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "feature": args.feature,
        "matrix_shape": list(x.shape),
        "pairchem_start": pair_start,
        "n_pairchem_features": n_pair,
        "selection_mode": "base_plus_pair" if args.pair_top_k > 0 else "global_topk",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nSelection")
    print(selection.to_string(index=False))
    print(f"\nSaved outputs to {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature", default="cheme_2d_full_boltz_log2fc_pred_seed10ens"
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_ROOT / "binding_seed42_top500_v2_6"
    )
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument(
        "--pair-top-k",
        type=int,
        default=0,
        help="If >0, select top-k base features plus this many pairchem features.",
    )
    parser.add_argument("--rank-n-estimators", type=int, default=500)
    parser.add_argument(
        "--tabpfn-version", choices=["v3", "v2_6", "v2_5", "v2"], default="v2_6"
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-before-softmax", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=255)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_oof(parse_args())


if __name__ == "__main__":
    main()
