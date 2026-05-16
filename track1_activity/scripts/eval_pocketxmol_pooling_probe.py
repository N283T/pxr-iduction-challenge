"""Quick CV bake-off for pooled PocketXMol hidden-state features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "track1_activity" / "src"))

from data import get_engine, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="+", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--max-k", type=int, default=512)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--models", nargs="+", default=["ridge", "topk_ridge"])
    return parser.parse_args()


def load_train_compound_ids() -> list[int]:
    sql = "SELECT compound_id FROM train_activity ORDER BY id"
    df = pd.read_sql(sql, get_engine())
    return df["compound_id"].astype(int).tolist()


def load_feature_npz(
    path: Path, train_compound_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    data = np.load(path, allow_pickle=True)
    compound_ids = data["compound_id"].astype(int)
    features = data["embedding"].astype(np.float32, copy=False)
    row_by_cid = {int(cid): i for i, cid in enumerate(compound_ids)}
    covered_positions = np.array(
        [i for i, cid in enumerate(train_compound_ids) if cid in row_by_cid],
        dtype=np.int64,
    )
    missing = [cid for cid in train_compound_ids if cid not in row_by_cid]
    indices = np.array(
        [row_by_cid[train_compound_ids[i]] for i in covered_positions],
        dtype=np.int64,
    )
    return features[indices], covered_positions, missing


def make_model(name: str, n_features: int, args: argparse.Namespace):
    alphas = np.logspace(-4, 4, 17)
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=alphas),
        )
    if name == "topk_ridge":
        k = min(args.max_k, n_features)
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SelectKBest(f_regression, k=k),
            RidgeCV(alphas=alphas),
        )
    if name == "pca_ridge":
        k = min(args.pca_components, n_features)
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=k, random_state=42),
            RidgeCV(alphas=alphas),
        )
    if name == "pca_hgb":
        k = min(args.pca_components, n_features)
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=k, random_state=42),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.04,
                max_iter=400,
                l2_regularization=0.01,
                random_state=42,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        model = make_model(model_name, X.shape[1], args)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx]).astype(np.float32)
        oof[val_idx] = pred
        fold_metric = compute_metrics(y[val_idx], pred)
        fold_metric["fold"] = fold
        fold_rows.append(fold_metric)
        print(
            f"    fold {fold}: MAE={fold_metric['MAE']:.5f} "
            f"Spearman={fold_metric['Spearman_R']:.5f}",
            flush=True,
        )
    if np.isnan(oof).any():
        raise ValueError("OOF has NaNs")
    return compute_metrics(y, oof), fold_rows


def main() -> None:
    args = parse_args()
    train_df = load_train_smiles_target()
    smiles = train_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    train_compound_ids = load_train_compound_ids()
    split_cache: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    rows = []

    for npz_path in args.npz:
        X, covered_positions, missing = load_feature_npz(npz_path, train_compound_ids)
        y_cov = y[covered_positions]
        smiles_cov = [smiles[i] for i in covered_positions]
        cache_key = tuple(int(i) for i in covered_positions)
        if cache_key not in split_cache:
            split_cache[cache_key] = umap_split_indices(
                smiles_cov, n_splits=5, n_clusters=50, seed=42
            )
        splits = split_cache[cache_key]
        print(f"\n=== {npz_path.name}: {X.shape[0]} x {X.shape[1]} ===", flush=True)
        if missing:
            print(f"  coverage: skipped {len(missing)} train compounds", flush=True)
        for model_name in args.models:
            print(f"  {model_name}", flush=True)
            metrics, fold_metrics = run_cv(X, y_cov, splits, model_name, args)
            row = {
                "feature": npz_path.stem,
                "model": model_name,
                "n_features": X.shape[1],
                **metrics,
                "fold_mae_std": float(np.std([m["MAE"] for m in fold_metrics])),
            }
            rows.append(row)
            print(
                f"  => MAE={metrics['MAE']:.5f} RAE={metrics['RAE']:.5f} "
                f"R2={metrics['R2']:.5f} Spearman={metrics['Spearman_R']:.5f}",
                flush=True,
            )

    result = pd.DataFrame(rows).sort_values(["MAE", "RAE"])
    print("\n=== Summary ===")
    print(result.to_markdown(index=False, floatfmt=".5f"))
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out_csv, index=False)
        print(args.out_csv)


if __name__ == "__main__":
    main()
