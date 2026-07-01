#!/usr/bin/env -S pixi run python
"""Sweep TabFM feature counts on Phase-1-style AS1 replay.

This reuses one feature load, one LGBM feature ranking, and one TabFM checkpoint
load, then fits train_activity-only TabFM regressors for a list of top-k feature
counts and evaluates the resulting test predictions on released AS1 labels.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_tabfm_probe as probe
import run_train
from data import load_test_smiles, load_train_smiles_target
from evaluate import compute_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "track1_activity" / "analysis" / "tabfm_topk_sweep" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature",
        default="cheme_2d_full_boltz_log2fc_pred_seed10ens",
        help="Feature set understood by run_train.load_features.",
    )
    parser.add_argument(
        "--top-k-list",
        default="64,128,256,500,1000,all",
        help="Comma-separated feature counts. Use 'all' or <=0 for all columns.",
    )
    parser.add_argument(
        "--tabfm-repo",
        type=Path,
        default=None,
        help="Path to a google-research/tabfm clone. Added to sys.path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--norm-methods", default="none")
    parser.add_argument(
        "--feat-shuffle-method",
        default="none",
        choices=["random", "none"],
    )
    parser.add_argument("--max-num-rows", type=int, default=4096)
    parser.add_argument("--max-num-features", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N entries from --top-k-list.",
    )
    parser.add_argument(
        "--write-submissions",
        action="store_true",
        help="Also write Track 1 submission-shaped CSVs under ignored submissions/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip matching top-k rows already present in the summary CSV.",
    )
    return parser.parse_args()


def parse_top_k_list(text: str, n_features: int) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item == "all":
            value = n_features
        else:
            value = int(item)
            if value <= 0 or value >= n_features:
                value = n_features
        if value not in values:
            values.append(value)
    return values


def rank_features(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    print("\nFitting one LGBM feature ranker ...")
    selector = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    selector.fit(X_train, y_train)
    gain = selector.booster_.feature_importance(importance_type="gain")
    nonzero = int(np.sum(gain > 0))
    print(f"  ranked {X_train.shape[1]} columns; {nonzero} have non-zero gain")
    return np.argsort(-gain)


def make_regressor_args(args: argparse.Namespace, top_k: int) -> argparse.Namespace:
    return argparse.Namespace(
        n_estimators=args.n_estimators,
        norm_methods=args.norm_methods,
        feat_shuffle_method=args.feat_shuffle_method,
        max_num_features=args.max_num_features,
        max_num_rows=args.max_num_rows,
        batch_size=args.batch_size,
        seed=args.seed,
        use_ensemble_preset=False,
    )


def load_existing_rows(args: argparse.Namespace) -> tuple[list[dict], set[int]]:
    summary_path = OUT_DIR / "tabfm_topk_sweep_summary.csv"
    if not args.resume or not summary_path.exists():
        return [], set()
    df = pd.read_csv(summary_path)
    if df.empty:
        return [], set()
    mask = (
        (df["feature"] == args.feature)
        & (df["n_estimators"] == args.n_estimators)
        & (df["norm_methods"] == args.norm_methods)
        & (df["feat_shuffle_method"] == args.feat_shuffle_method)
        & (df["max_num_rows"] == args.max_num_rows)
        & (df["batch_size"] == args.batch_size)
        & (df["seed"] == args.seed)
    )
    if args.max_num_features is None:
        mask &= df["max_num_features"].isna()
    else:
        mask &= df["max_num_features"] == args.max_num_features
    existing = df.loc[mask].copy()
    if existing.empty:
        return df.to_dict("records"), set()
    skip_top_k = {int(v) for v in existing["top_k"].tolist()}
    print(f"Resume: found existing matching top-k values: {sorted(skip_top_k)}")
    return df.to_dict("records"), skip_top_k


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tabfm = probe._import_tabfm(args.tabfm_repo)
    print(f"Using tabfm {getattr(tabfm, '__version__', '?')}")

    print("Loading Track 1 data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    print(f"\nLoading feature set: {args.feature}")
    X_train, X_test = run_train.load_features(args.feature, train_df, test_df)
    print(f"  raw shapes: train {X_train.shape}, test {X_test.shape}")
    X_train, X_test = probe._sanitize_like_tabpfn(X_train, X_test)

    order = rank_features(X_train, y_train)
    top_k_values = parse_top_k_list(args.top_k_list, X_train.shape[1])
    if args.limit is not None:
        top_k_values = top_k_values[: args.limit]
    print(f"\nSweep top-k values: {top_k_values}")

    print("\nLoading TabFM PyTorch regression weights ...")
    model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression", device=args.device)

    as1 = probe._load_as1_labels()
    as1_idx = as1["test_idx"].to_numpy(dtype=int)
    as1_y = as1["as1_pec50"].to_numpy(dtype=np.float64)

    rows, skip_top_k = load_existing_rows(args)
    for top_k in top_k_values:
        if top_k in skip_top_k:
            print(f"\n[top-{top_k}] skipping existing result")
            continue
        started = time.time()
        selected = order[:top_k]
        print(f"\n[top-{top_k}] fitting TabFM ...")
        X_train_df = probe._as_frame(X_train[:, selected])
        X_test_df = probe._as_frame(X_test[:, selected])

        reg = probe._make_regressor(tabfm, model, make_regressor_args(args, top_k))
        reg.fit(X_train_df, y_train)
        test_pred = np.asarray(reg.predict(X_test_df), dtype=np.float64)
        as1_pred = test_pred[as1_idx]
        metrics = compute_metrics(as1_y, as1_pred)
        bias = float(np.mean(as1_pred - as1_y))
        elapsed_s = time.time() - started
        print(
            f"  AS1 MAE={metrics['MAE']:.4f} RAE={metrics['RAE']:.4f} "
            f"Sp={metrics['Spearman_R']:.4f} bias={bias:+.4f} "
            f"elapsed={elapsed_s / 60:.1f}m"
        )

        norm_slug = args.norm_methods.replace(",", "-").replace("/", "_")
        shuffle_slug = args.feat_shuffle_method.replace("/", "_")
        pred_path = OUT_DIR / (
            f"tabfm_top{top_k}_ne{args.n_estimators}_{norm_slug}_"
            f"{shuffle_slug}_mr{args.max_num_rows or 'all'}_test_predictions.csv"
        )
        pd.DataFrame(
            {
                "SMILES": test_df["smiles"],
                "Molecule Name": test_df["molecule_name"],
                "pEC50": test_pred,
            }
        ).to_csv(pred_path, index=False)

        if args.write_submissions:
            sub_name = (
                f"tabfm_phase1_replay_{args.feature}_top{top_k}"
                f"_ne{args.n_estimators}_mr{args.max_num_rows or 'all'}.csv"
            )
            sub_path = probe.SUBMISSION_DIR / sub_name
            sub_path.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(pred_path).to_csv(sub_path, index=False)
        else:
            sub_path = None

        rows.append(
            {
                "feature": args.feature,
                "top_k": int(top_k),
                "n_estimators": int(args.n_estimators),
                "norm_methods": args.norm_methods,
                "feat_shuffle_method": args.feat_shuffle_method,
                "max_num_rows": args.max_num_rows,
                "max_num_features": args.max_num_features,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "as1_n": int(len(as1_y)),
                "as1_mae": float(metrics["MAE"]),
                "as1_rae": float(metrics["RAE"]),
                "as1_spearman": float(metrics["Spearman_R"]),
                "as1_r2": float(metrics["R2"]),
                "as1_bias": bias,
                "test_mean": float(test_pred.mean()),
                "test_std": float(test_pred.std()),
                "test_min": float(test_pred.min()),
                "test_max": float(test_pred.max()),
                "elapsed_s": float(elapsed_s),
                "prediction_path": str(pred_path.relative_to(REPO_ROOT)),
                "submission_path": (
                    str(sub_path.relative_to(REPO_ROOT)) if sub_path else None
                ),
            }
        )
        pd.DataFrame(rows).sort_values("as1_mae").to_csv(
            OUT_DIR / "tabfm_topk_sweep_summary.csv", index=False
        )
        del reg, X_train_df, X_test_df
        probe._clear_cuda()

    summary = pd.DataFrame(rows).sort_values("as1_mae")
    summary_path = OUT_DIR / "tabfm_topk_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary[["top_k", "as1_mae", "as1_spearman", "as1_bias", "elapsed_s"]])


if __name__ == "__main__":
    main()
