"""Train a fold-safe ROCS prototype axis from OpenEye FastROCS scores.

This first variant reuses ``compound_rocs.all_query_scores``. That table scores
all train/test compounds against the potent train prototypes computed by
``compute_rocs_to_actives.py``. OOF fold-safety is restored by selecting only
query prototypes that belong to the current training fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from rocs_prototype import build_prototype_features, complete_score_maps  # noqa: E402
from run_train import _build_umap_split_features, load_compound_ids, load_features  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def _rows_to_score_maps(
    compound_ids: list[int], df: pd.DataFrame, source_name: str
) -> dict[int, dict[str, list[float]]]:
    out: dict[int, dict[str, list[float]]] = {}
    for row in df.itertuples(index=False):
        value = row.all_query_scores
        if isinstance(value, str):
            value = json.loads(value)
        out[int(row.compound_id)] = value or {}
    missing = sorted(set(compound_ids) - set(out))
    if missing:
        print(
            f"  {source_name} missing {len(missing)} compounds; "
            "using empty ROCS scores for uncovered Omega failures"
        )
    return complete_score_maps(compound_ids, out)


def load_rocs_score_maps(compound_ids: list[int]) -> dict[int, dict[str, list[float]]]:
    """Load per-target active-query score maps from ``compound_rocs``."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        ph = ",".join(["%s"] * len(compound_ids))
        df = pd.read_sql(
            f"SELECT compound_id, all_query_scores FROM compound_rocs WHERE compound_id IN ({ph})",
            conn,
            params=compound_ids,
        )
    return _rows_to_score_maps(compound_ids, df, "compound_rocs")


def load_score_maps_parquet(
    path: Path, compound_ids: list[int]
) -> dict[int, dict[str, list[float]]]:
    df = pd.read_parquet(path)
    return _rows_to_score_maps(compound_ids, df, str(path))


def available_query_ids(score_maps: dict[int, dict[str, list[float]]]) -> set[int]:
    ids: set[int] = set()
    for score_map in score_maps.values():
        ids.update(int(k) for k in score_map.keys())
    return ids


def select_queries_by_activity(
    compound_ids: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    available_queries: set[int],
    n_queries: int,
    high: bool,
) -> list[int]:
    """Select high- or low-pEC50 query IDs available in a ROCS score table."""
    fold_ids = compound_ids[train_idx]
    fold_y = y[train_idx]
    order = np.argsort(-fold_y if high else fold_y, kind="stable")
    selected: list[int] = []
    for i in order:
        cid = int(fold_ids[i])
        if cid in available_queries:
            selected.append(cid)
        if len(selected) >= n_queries:
            break
    if not selected:
        label = "active" if high else "inactive"
        raise ValueError(f"No {label} query prototypes available for this fold")
    return selected


def make_feature_matrix(
    target_ids: list[int] | np.ndarray,
    active_maps: dict[int, dict[str, list[float]]],
    active_query_ids: list[int],
    query_targets: dict[int, float],
    inactive_maps: dict[int, dict[str, list[float]]] | None = None,
    inactive_query_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    target_ids = [int(x) for x in target_ids]
    active_X, active_names = build_prototype_features(
        target_ids=target_ids,
        score_maps=active_maps,
        query_ids=active_query_ids,
        query_targets=query_targets,
        prefix="active_rocs",
    )
    if inactive_maps is None or inactive_query_ids is None:
        return active_X, active_names
    inactive_X, inactive_names = build_prototype_features(
        target_ids=target_ids,
        score_maps=inactive_maps,
        query_ids=inactive_query_ids,
        query_targets=query_targets,
        prefix="inactive_rocs",
    )
    delta_X = active_X - inactive_X
    delta_names = [name.replace("active_rocs_", "delta_rocs_") for name in active_names]
    return (
        np.concatenate([active_X, inactive_X, delta_X], axis=1).astype(np.float32),
        active_names + inactive_names + delta_names,
    )


def concat_base_features(
    base_name: str,
    base_train: np.ndarray | None,
    base_test: np.ndarray | None,
    train_idx: np.ndarray | None,
    rocs_X: np.ndarray,
    is_test: bool = False,
) -> np.ndarray:
    if base_name == "none":
        return rocs_X
    if is_test:
        assert base_test is not None
        return np.concatenate([base_test, rocs_X], axis=1).astype(np.float32)
    assert base_train is not None and train_idx is not None
    return np.concatenate([base_train[train_idx], rocs_X], axis=1).astype(np.float32)


def fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def run(args: argparse.Namespace) -> dict:
    print(
        f"ROCS prototype axis: n_active={args.n_active}, n_inactive={args.n_inactive}, "
        f"base={args.base_feature}, inactive={bool(args.inactive_score_parquet)}, "
        f"umap_seed={args.umap_seed}, clusters={args.umap_clusters}"
    )
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    train_ids = np.asarray(load_compound_ids("train"), dtype=int)
    test_ids = np.asarray(load_compound_ids("test"), dtype=int)
    query_targets = {int(cid): float(val) for cid, val in zip(train_ids, y, strict=True)}

    all_ids = [int(x) for x in np.concatenate([train_ids, test_ids])]
    print("Loading compound_rocs active all_query_scores...")
    score_maps = load_rocs_score_maps(all_ids)
    available = available_query_ids(score_maps)
    print(f"  available active query prototypes: {len(available)}")
    inactive_maps = None
    inactive_available: set[int] = set()
    if args.inactive_score_parquet is not None:
        print(f"Loading inactive ROCS score parquet: {args.inactive_score_parquet}")
        inactive_maps = load_score_maps_parquet(args.inactive_score_parquet, all_ids)
        inactive_available = available_query_ids(inactive_maps)
        print(f"  available inactive query prototypes: {len(inactive_available)}")

    base_train = base_test = None
    if args.base_feature != "none":
        base_train, base_test = load_features(args.base_feature, train_df, test_df)
        base_train = base_train.astype(np.float32)
        base_test = base_test.astype(np.float32)
        print(f"  base features: train={base_train.shape}, test={base_test.shape}")

    split_features, split_metric = _build_umap_split_features("morgan", train_ids.tolist())
    splits = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=5,
        n_clusters=args.umap_clusters,
        seed=args.umap_seed,
        features=split_features,
        metric=split_metric,
    )

    oof = np.zeros(len(train_ids), dtype=np.float32)
    fold_metrics: list[dict] = []
    feature_names: list[str] | None = None
    for fold, (tr_idx, va_idx) in enumerate(splits):
        query_ids = select_queries_by_activity(
            train_ids, y, tr_idx, available, args.n_active, high=True
        )
        inactive_query_ids = None
        if inactive_maps is not None:
            inactive_query_ids = select_queries_by_activity(
                train_ids, y, tr_idx, inactive_available, args.n_inactive, high=False
            )
        X_rocs_tr, names = make_feature_matrix(
            train_ids[tr_idx],
            score_maps,
            query_ids,
            query_targets,
            inactive_maps,
            inactive_query_ids,
        )
        X_rocs_va, _ = make_feature_matrix(
            train_ids[va_idx],
            score_maps,
            query_ids,
            query_targets,
            inactive_maps,
            inactive_query_ids,
        )
        feature_names = names
        X_tr = concat_base_features(args.base_feature, base_train, base_test, tr_idx, X_rocs_tr)
        X_va = concat_base_features(args.base_feature, base_train, base_test, va_idx, X_rocs_va)
        model = fit_lgbm(X_tr, y[tr_idx], seed=args.seed + fold)
        pred = model.predict(X_va).astype(np.float32)
        oof[va_idx] = pred
        metrics = compute_metrics(y[va_idx], pred)
        fold_metrics.append(metrics)
        inactive_label = f"/{len(inactive_query_ids)}" if inactive_query_ids is not None else ""
        print_metrics(metrics, label=f"Fold {fold} q={len(query_ids)}{inactive_label}")

    print("\nOverall OOF:")
    oof_metrics = compute_metrics(y, oof)
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    final_queries = select_queries_by_activity(
        train_ids, y, np.arange(len(train_ids)), available, args.n_active, high=True
    )
    final_inactive_queries = None
    if inactive_maps is not None:
        final_inactive_queries = select_queries_by_activity(
            train_ids,
            y,
            np.arange(len(train_ids)),
            inactive_available,
            args.n_inactive,
            high=False,
        )
    X_rocs_all, final_names = make_feature_matrix(
        train_ids,
        score_maps,
        final_queries,
        query_targets,
        inactive_maps,
        final_inactive_queries,
    )
    X_rocs_test, _ = make_feature_matrix(
        test_ids,
        score_maps,
        final_queries,
        query_targets,
        inactive_maps,
        final_inactive_queries,
    )
    X_all = concat_base_features(
        args.base_feature, base_train, base_test, np.arange(len(train_ids)), X_rocs_all
    )
    X_test = concat_base_features(
        args.base_feature, base_train, base_test, None, X_rocs_test, is_test=True
    )
    final_model = fit_lgbm(X_all, y, seed=args.seed)
    test_preds = final_model.predict(X_test).astype(np.float32)
    print(f"  Test preds: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    contrast_tag = f"_inactive_n{args.n_inactive}" if args.inactive_score_parquet is not None else ""
    exp_name = f"lgbm_rocs_active_proto_n{args.n_active}{contrast_tag}_{args.base_feature}_umap"
    if args.umap_seed != 42:
        exp_name += f"_s{args.umap_seed}"
    if args.umap_clusters != 50:
        exp_name += f"_k{args.umap_clusters}"
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    exp_id = record_experiment(
        name=exp_name,
        description="Fold-safe active-prototype ROCS summaries from OpenEye FastROCS scores",
        model_type="lgbm",
        feature_set=f"rocs_active_proto+{args.base_feature}",
        hyperparameters={
            "n_active": args.n_active,
            "n_inactive": args.n_inactive if args.inactive_score_parquet is not None else 0,
            "base_feature": args.base_feature,
            "feature_names": final_names if feature_names is None else feature_names,
            "source_table": "compound_rocs",
            "inactive_score_parquet": str(args.inactive_score_parquet) if args.inactive_score_parquet else None,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}; fold-safe active query filtering",
        on_conflict_replace=args.replace,
    )
    save_oof_predictions(exp_id, oof)
    print(f"\nDone: {exp_name} id={exp_id} RAE={oof_metrics['RAE']:.4f}")
    return oof_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-active", type=int, default=64)
    parser.add_argument("--n-inactive", type=int, default=64)
    parser.add_argument("--inactive-score-parquet", type=Path)
    parser.add_argument("--base-feature", default="none")
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--umap-clusters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
