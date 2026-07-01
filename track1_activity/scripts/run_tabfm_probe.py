#!/usr/bin/env -S pixi run python
"""Probe Google TabFM on Track 1 tabular features.

TabFM is not vendored in this repository. Install it in the pixi environment
or pass ``--tabfm-repo`` pointing at a clone of google-research/tabfm, e.g.:

    ghq get google-research/tabfm
    pixi run python -m pip install --no-deps absl-py 'jaxtyping<0.3' 'typeguard<3'
    pixi run python track1_activity/scripts/run_tabfm_probe.py \
        --tabfm-repo "$(ghq root)/github.com/google-research/tabfm" \
        --fold-limit 1 --top-k 128 --max-num-rows 1024 --n-estimators 1

The PyTorch TabFM weights are released separately on Hugging Face under the
tabfm-non-commercial-v1.0 license. Treat outputs as research probes unless the
license is appropriate for the intended use.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics, record_experiment, save_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
EXPERIMENT_PREFIX = "tabfm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature",
        default="cheme_2d_full_boltz_log2fc_pred_seed10ens",
        help="Feature set understood by run_train.load_features.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Select top K features by full-train LGBM gain. Use <=0 for all.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=None,
        help="Run only the first N CV folds for a quick probe.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip full-train fit and test CSV generation.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip train OOF CV. Useful for Phase-1-style AS1 replay.",
    )
    parser.add_argument(
        "--eval-as1",
        action="store_true",
        help="Evaluate test predictions on released Phase 1/AS1 labels.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record experiment and OOF predictions in the DB. Requires full CV.",
    )
    parser.add_argument(
        "--on-conflict-replace",
        action="store_true",
        help="Replace an existing experiment with the same name when recording.",
    )
    parser.add_argument(
        "--tabfm-repo",
        type=Path,
        default=None,
        help="Path to a google-research/tabfm clone. Added to sys.path.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to tabfm_v1_0_0_pytorch.load, e.g. cuda or cpu.",
    )
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument(
        "--norm-methods",
        default="none,power",
        help="Comma-separated TabFM normalization methods.",
    )
    parser.add_argument(
        "--feat-shuffle-method",
        default="random",
        choices=["random", "none"],
    )
    parser.add_argument(
        "--max-num-rows",
        type=int,
        default=4096,
        help="TabFM context rows per ensemble member. Use <=0 for no cap.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=None,
        help="TabFM feature cap after this script's top-k selection.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--use-ensemble-preset",
        action="store_true",
        help="Use TabFMRegressor.ensemble; incompatible with max-num-rows.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the experiment/submission name.",
    )
    return parser.parse_args()


def _import_tabfm(tabfm_repo: Path | None):
    if tabfm_repo is not None:
        sys.path.insert(0, str(tabfm_repo.resolve()))
    try:
        import tabfm  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "Could not import tabfm. Install google-research/tabfm or pass "
            "--tabfm-repo /path/to/google-research/tabfm. Minimal deps used "
            "here: absl-py, jaxtyping<0.3, typeguard<3."
        ) from exc
    return tabfm


def _sanitize_like_tabpfn(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.astype(np.float64, copy=True)
    X_test = X_test.astype(np.float64, copy=True)
    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, col_mean)
    X_test = np.where(np.isfinite(X_test), X_test, col_mean)
    return X_train, X_test


def _select_top_k(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_k <= 0 or top_k >= X_train.shape[1]:
        return X_train, X_test, np.arange(X_train.shape[1])

    print(f"\nFitting LGBM feature selector for top-{top_k} ...")
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
    selected = np.argsort(-gain)[:top_k]
    nonzero = int(np.sum(gain[selected] > 0))
    print(f"  selected {len(selected)} columns; {nonzero} have non-zero gain")
    return X_train[:, selected], X_test[:, selected], selected


def _as_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=[f"f{i:04d}" for i in range(X.shape[1])])


def _load_as1_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            ordered_test.test_idx,
            ordered_test.molecule_name,
            ordered_test.smiles,
            l.pec50 AS as1_pec50
        FROM (
            SELECT
                row_number() OVER (ORDER BY t.id) - 1 AS test_idx,
                t.compound_id,
                c.molecule_name,
                c.std_smiles AS smiles
            FROM test_activity t
            JOIN compounds c ON c.id = t.compound_id
        ) ordered_test
        JOIN test_activity_phase1_labels l
          ON l.compound_id = ordered_test.compound_id
        ORDER BY ordered_test.test_idx
        """,
        get_engine(),
    )


def _make_regressor(tabfm, model, args: argparse.Namespace):
    norm_methods = [m.strip() for m in args.norm_methods.split(",") if m.strip()]
    max_num_rows = (
        args.max_num_rows if args.max_num_rows and args.max_num_rows > 0 else None
    )
    max_num_features = (
        args.max_num_features
        if args.max_num_features is not None and args.max_num_features > 0
        else None
    )
    common = {
        "model": model,
        "n_estimators": args.n_estimators,
        "norm_methods": norm_methods,
        "feat_shuffle_method": args.feat_shuffle_method,
        "max_num_features": max_num_features,
        "max_num_rows": max_num_rows,
        "batch_size": args.batch_size,
        "random_state": args.seed,
    }
    if args.use_ensemble_preset:
        if max_num_rows is not None:
            raise SystemExit(
                "--use-ensemble-preset cannot be combined with --max-num-rows"
            )
        return tabfm.TabFMRegressor.ensemble(**common)
    return tabfm.TabFMRegressor(**common)


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    if args.record and args.fold_limit is not None:
        raise SystemExit("--record requires full CV; remove --fold-limit.")
    if args.record and args.skip_cv:
        raise SystemExit("--record requires CV OOF predictions; remove --skip-cv.")
    if args.eval_as1 and args.skip_test:
        raise SystemExit("--eval-as1 requires test predictions; remove --skip-test.")

    tabfm = _import_tabfm(args.tabfm_repo)
    print(f"Using tabfm {getattr(tabfm, '__version__', '?')}")

    print("Loading Track 1 data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    print(f"\nLoading feature set: {args.feature}")
    X_train, X_test = run_train.load_features(args.feature, train_df, test_df)
    print(f"  raw shapes: train {X_train.shape}, test {X_test.shape}")
    X_train, X_test = _sanitize_like_tabpfn(X_train, X_test)
    X_train, X_test, selected = _select_top_k(X_train, y_train, X_test, args.top_k)
    print(f"  model shapes: train {X_train.shape}, test {X_test.shape}")

    X_train_df = _as_frame(X_train)
    X_test_df = _as_frame(X_test)

    print("\nLoading TabFM PyTorch regression weights ...")
    model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression", device=args.device)

    oof_preds = np.full(len(X_train_df), np.nan, dtype=np.float64)
    covered_mask = np.zeros(len(X_train_df), dtype=bool)
    fold_metrics: list[dict] = []
    overall = None
    if not args.skip_cv:
        print(
            f"\nUMAP {args.n_splits}-fold split "
            f"(Morgan+Jaccard, k={args.n_clusters}, seed={args.seed}) ..."
        )
        folds = umap_split_indices(
            train_df["smiles"].tolist(),
            n_splits=args.n_splits,
            n_clusters=args.n_clusters,
            seed=args.seed,
        )
        if args.fold_limit is not None:
            folds = folds[: args.fold_limit]

        print("\nCross-validating TabFM ...")
        for fold, (tr_idx, va_idx) in enumerate(folds):
            reg = _make_regressor(tabfm, model, args)
            reg.fit(X_train_df.iloc[tr_idx], y_train[tr_idx])
            pred = np.asarray(reg.predict(X_train_df.iloc[va_idx]), dtype=np.float64)
            oof_preds[va_idx] = pred
            covered_mask[va_idx] = True
            metrics = compute_metrics(y_train[va_idx], pred)
            fold_metrics.append(metrics)
            print(
                f"  fold {fold}: train={len(tr_idx)} val={len(va_idx)} "
                f"MAE={metrics['MAE']:.4f} RAE={metrics['RAE']:.4f} "
                f"Sp={metrics['Spearman_R']:.4f}"
            )
            del reg
            _clear_cuda()

        covered = int(covered_mask.sum())
        overall = compute_metrics(y_train[covered_mask], oof_preds[covered_mask])
        label = (
            "full OOF"
            if covered == len(y_train)
            else f"partial OOF ({covered}/{len(y_train)})"
        )
        print(
            f"\n  {label}: MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f} "
            f"Sp={overall['Spearman_R']:.4f} R2={overall['R2']:.4f}"
        )

    default_name = (
        f"{EXPERIMENT_PREFIX}_{args.feature}_top{len(selected)}"
        f"_ne{args.n_estimators}_mr{args.max_num_rows or 'all'}_umap"
    )
    experiment_name = args.experiment_name or default_name

    sub_path = None
    test_preds = None
    if not args.skip_test:
        print("\nFitting on ALL train for test prediction ...")
        reg = _make_regressor(tabfm, model, args)
        reg.fit(X_train_df, y_train)
        test_preds = np.asarray(reg.predict(X_test_df), dtype=np.float64)
        print(
            f"  test preds: mean={test_preds.mean():.4f} std={test_preds.std():.4f} "
            f"min={test_preds.min():.4f} max={test_preds.max():.4f}"
        )
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        sub = pd.DataFrame(
            {
                "SMILES": test_df["smiles"],
                "Molecule Name": test_df["molecule_name"],
                "pEC50": test_preds,
            }
        )
        sub_path = SUBMISSION_DIR.joinpath(f"{experiment_name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f"  wrote {sub_path}")
        del reg
        _clear_cuda()

    if args.eval_as1:
        if test_preds is None:
            raise RuntimeError("Internal error: AS1 eval requested without test preds")
        as1 = _load_as1_labels()
        as1_idx = as1["test_idx"].to_numpy(dtype=int)
        as1_y = as1["as1_pec50"].to_numpy(dtype=np.float64)
        as1_pred = test_preds[as1_idx]
        as1_metrics = compute_metrics(as1_y, as1_pred)
        bias = float(np.mean(as1_pred - as1_y))
        print("\nAS1 replay (train_activity only -> released Phase 1 test labels):")
        print(
            f"  n={len(as1)} MAE={as1_metrics['MAE']:.4f} "
            f"RAE={as1_metrics['RAE']:.4f} Sp={as1_metrics['Spearman_R']:.4f} "
            f"R2={as1_metrics['R2']:.4f} bias={bias:+.4f}"
        )

    if args.record:
        print("\nRecording experiment in DB ...")
        if overall is None:
            raise RuntimeError(
                "Internal error: recording requested without OOF metrics"
            )
        exp_id = record_experiment(
            name=experiment_name,
            description=(
                f"Google TabFM v1.0.0 PyTorch regression on top-{len(selected)} "
                f"columns from {args.feature}; zero-shot/in-context fit."
            ),
            model_type="tabfm",
            feature_set=f"{args.feature}_top{len(selected)}",
            hyperparameters={
                "feature": args.feature,
                "top_k": args.top_k,
                "selected_columns": selected.astype(int).tolist(),
                "n_estimators": args.n_estimators,
                "norm_methods": args.norm_methods,
                "feat_shuffle_method": args.feat_shuffle_method,
                "max_num_rows": args.max_num_rows,
                "max_num_features": args.max_num_features,
                "batch_size": args.batch_size,
                "device": args.device,
                "n_splits": args.n_splits,
                "n_clusters": args.n_clusters,
                "seed": args.seed,
                "use_ensemble_preset": args.use_ensemble_preset,
                "license_note": "HF weights: tabfm-non-commercial-v1.0",
            },
            fold_metrics=fold_metrics,
            submission_path=(
                str(sub_path.relative_to(REPO_ROOT)) if sub_path is not None else None
            ),
            notes=(
                f"OOF MAE={overall['MAE']:.4f}; TabFM v1.0.0 PyTorch "
                "research probe. Weight license is non-commercial."
            ),
            on_conflict_replace=args.on_conflict_replace,
        )
        save_oof_predictions(exp_id, oof_preds)
        print(f"  recorded experiment id={exp_id}")


if __name__ == "__main__":
    main()
