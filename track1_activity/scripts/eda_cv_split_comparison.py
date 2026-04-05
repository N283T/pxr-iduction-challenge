#!/usr/bin/env -S pixi run python
"""Compare CV split strategies: Random vs Murcko Scaffold vs Butina clustering.

Uses a fixed LightGBM model (best hyperparams from single_mordred experiment)
to isolate the effect of the split strategy on CV RAE.

Addresses issues #17 and #20.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import lightgbm as lgb
import numpy as np
from collections import defaultdict
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import KFold

import umap

from data import load_train_smiles_target, load_train_mordred
from evaluate import compute_metrics, print_metrics


# Fixed hyperparams from best single model (experiment id=59)
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "verbose": -1,
    "seed": 42,
    "num_leaves": 84,
    "learning_rate": 0.0165,
    "feature_fraction": 0.626,
    "bagging_fraction": 0.903,
    "bagging_freq": 3,
    "min_child_samples": 81,
    "lambda_l1": 1.19e-5,
    "lambda_l2": 0.598,
}


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def random_split(n: int, n_splits: int = 5, seed: int = 42):
    """Standard random K-Fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def scaffold_split(smiles_list: list[str], n_splits: int = 5, seed: int = 42):
    """Murcko scaffold split (existing implementation)."""
    from splits import scaffold_split_indices

    return scaffold_split_indices(smiles_list, n_splits=n_splits, seed=seed)


def butina_split(
    smiles_list: list[str],
    n_splits: int = 5,
    cutoff: float = 0.6,
    seed: int = 42,
):
    """Butina cluster-based split."""
    rng = np.random.RandomState(seed)

    # Compute Morgan fingerprints
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol))

    n = len(fps)

    # Butina clustering
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Sort clusters by size (largest first)
    cluster_list = sorted(clusters, key=len, reverse=True)

    # Assign clusters to folds (greedy, like scaffold split)
    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for cluster in cluster_list:
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(cluster)
        fold_sizes[min_fold] += len(cluster)

    for indices in fold_indices:
        rng.shuffle(indices)

    splits = []
    for val_fold in range(n_splits):
        val_idx = np.array(fold_indices[val_fold])
        train_idx = np.concatenate(
            [np.array(fold_indices[i]) for i in range(n_splits) if i != val_fold]
        )
        splits.append((train_idx, val_idx))

    return splits


def umap_split(
    smiles_list: list[str],
    n_splits: int = 5,
    n_clusters: int = 50,
    seed: int = 42,
):
    """UMAP projection + KMeans clustering split (strictest separation)."""
    from sklearn.cluster import KMeans

    rng = np.random.RandomState(seed)

    # Compute Morgan fingerprints as numpy
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps_np = np.array(
        [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles_list],
        dtype=np.float32,
    )

    # UMAP projection
    reducer = umap.UMAP(
        n_components=10, metric="jaccard", random_state=seed, n_neighbors=30
    )
    embedding = reducer.fit_transform(fps_np)

    # KMeans clustering on UMAP embedding
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(embedding)

    # Group indices by cluster
    cluster_to_indices = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        cluster_to_indices[cl].append(i)

    cluster_groups = list(cluster_to_indices.values())
    cluster_groups.sort(key=len, reverse=True)

    # Assign clusters to folds (greedy)
    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for group in cluster_groups:
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(group)
        fold_sizes[min_fold] += len(group)

    for indices in fold_indices:
        rng.shuffle(indices)

    splits = []
    for val_fold in range(n_splits):
        val_idx = np.array(fold_indices[val_fold])
        train_idx = np.concatenate(
            [np.array(fold_indices[i]) for i in range(n_splits) if i != val_fold]
        )
        splits.append((train_idx, val_idx))

    return splits


def compute_split_leakage(smiles_list, splits):
    """Measure Tanimoto similarity between train/val per fold (Sheridan metric)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol))

    fold_sims = []
    for train_idx, val_idx in splits:
        train_fps = [fps[i] for i in train_idx]
        val_sims = []
        for vi in val_idx:
            sims = DataStructs.BulkTanimotoSimilarity(fps[vi], train_fps)
            top5 = sorted(sims, reverse=True)[:5]
            val_sims.append(np.mean(top5))
        fold_sims.append(np.mean(val_sims))
    return fold_sims


# ---------------------------------------------------------------------------
# Run CV with a given split
# ---------------------------------------------------------------------------


def run_cv(X, y, splits, label: str):
    """Run LightGBM CV with given splits and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    oof_preds = np.full(len(y), np.nan)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)

        model = lgb.train(
            LGBM_PARAMS,
            dt,
            num_boost_round=2000,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        vp = model.predict(X_va)
        oof_preds[va_idx] = vp
        metrics = compute_metrics(y_va, vp)
        fold_metrics.append(metrics)
        print(
            f"  Fold {fold}: n_train={len(tr_idx)} n_val={len(va_idx)} "
            f"RAE={metrics['RAE']:.4f} MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}"
        )

    # Overall OOF
    valid_mask = ~np.isnan(oof_preds)
    oof_metrics = compute_metrics(y[valid_mask], oof_preds[valid_mask])

    print(
        f"\n  Overall OOF: RAE={oof_metrics['RAE']:.4f} MAE={oof_metrics['MAE']:.4f} "
        f"R2={oof_metrics['R2']:.4f}"
    )

    # Per-fold stats
    raes = [m["RAE"] for m in fold_metrics]
    print(
        f"  Fold RAE: mean={np.mean(raes):.4f} ± {np.std(raes):.4f} "
        f"(min={min(raes):.4f} max={max(raes):.4f})"
    )

    return oof_metrics, fold_metrics, oof_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading data...")
    train_df = load_train_smiles_target()
    mordred_df, y_series = load_train_mordred()

    # Align mordred with train_df ordering
    # train_df is ordered by train_activity.id, mordred uses compound_id index
    # Need to match them
    smiles_list = train_df["smiles"].tolist()
    y = train_df["pec50"].values

    # Load mordred features aligned with train ordering
    from data import get_conn

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    train_cids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    X = mordred_df.loc[train_cids].values.astype(np.float32)
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Train: {X.shape[0]} compounds, {X.shape[1]} features")
    print(f"  Target: mean={y.mean():.3f} std={y.std():.3f}")

    # Define split strategies
    n = len(y)
    strategies = {
        "Random (5-fold)": random_split(n, n_splits=5, seed=42),
        "Murcko Scaffold (5-fold)": scaffold_split(smiles_list, n_splits=5, seed=42),
        "Butina cutoff=0.4 (5-fold)": butina_split(
            smiles_list, n_splits=5, cutoff=0.4, seed=42
        ),
        "Butina cutoff=0.5 (5-fold)": butina_split(
            smiles_list, n_splits=5, cutoff=0.5, seed=42
        ),
        "Butina cutoff=0.6 (5-fold)": butina_split(
            smiles_list, n_splits=5, cutoff=0.6, seed=42
        ),
        "Butina cutoff=0.7 (5-fold)": butina_split(
            smiles_list, n_splits=5, cutoff=0.7, seed=42
        ),
        "UMAP 50 clusters (5-fold)": umap_split(
            smiles_list, n_splits=5, n_clusters=50, seed=42
        ),
        "UMAP 100 clusters (5-fold)": umap_split(
            smiles_list, n_splits=5, n_clusters=100, seed=42
        ),
    }

    # Measure train-val similarity for each strategy
    print(f"\n{'=' * 60}")
    print("  TRAIN-VAL SIMILARITY BY SPLIT STRATEGY (5-NN Tanimoto)")
    print(f"{'=' * 60}")
    for name, splits in strategies.items():
        fold_sims = compute_split_leakage(smiles_list, splits)
        print(
            f"  {name:35s}: mean={np.mean(fold_sims):.4f} "
            f"(folds: {', '.join(f'{s:.4f}' for s in fold_sims)})"
        )

    # Run CV for each strategy
    results = {}
    for name, splits in strategies.items():
        oof_metrics, fold_metrics, oof_preds = run_cv(X, y, splits, name)
        results[name] = {
            "oof_metrics": oof_metrics,
            "fold_metrics": fold_metrics,
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: CV SPLIT COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<35s} {'OOF RAE':>8} {'Fold RAE std':>12} {'OOF R2':>8}")
    print(f"  {'-' * 65}")

    # LB RAE for reference
    lb_rae = 0.62
    for name in strategies:
        r = results[name]
        oof_rae = r["oof_metrics"]["RAE"]
        fold_raes = [m["RAE"] for m in r["fold_metrics"]]
        oof_r2 = r["oof_metrics"]["R2"]
        gap = lb_rae - oof_rae
        print(
            f"  {name:<35s} {oof_rae:8.4f} {np.std(fold_raes):12.4f} {oof_r2:8.4f}  "
            f"(LB gap: {gap:+.3f})"
        )

    print(f"\n  Leaderboard RAE (reference): {lb_rae}")
    print(f"\n  Key question: which split gives OOF RAE closest to LB RAE ({lb_rae})?")
    print(f"  The closer the match, the more realistic the CV estimate.")


if __name__ == "__main__":
    main()
