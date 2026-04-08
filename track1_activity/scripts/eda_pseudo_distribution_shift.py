#!/usr/bin/env -S pixi run python
"""Test the hypothesis that pseudo-label compounds are out of distribution
relative to train, as measured by the same UMAP space used for CV splits.

Approach:
1. Build Morgan FPs for train (4140), pseudo (8483 from data/pseudo_labels.parquet),
   and test (513) compounds.
2. Fit UMAP using the EXACT parameters from splits.umap_split_indices on train only.
3. Transform pseudo and test into the same UMAP space.
4. Cluster with KMeans (50 clusters) on train, assign pseudo/test to nearest cluster.
5. Quantify:
   a) For each pseudo compound, distance to its nearest train neighbor in UMAP space
   b) Cluster occupancy: how do pseudo compounds distribute across the 50 train clusters?
   c) Compare to test compounds (which should be the "real" OOD reference)
6. Save figure: 2D UMAP scatter (train vs pseudo vs test)
7. Print quantitative summary

Usage:
    pixi run python track1_activity/scripts/eda_pseudo_distribution_shift.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))
from data import get_engine  # noqa: E402

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
PSEUDO_PATH = Path("data/pseudo_labels.parquet")
SEED = 42
N_CLUSTERS = 50


def morgan_fps(smiles_list: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def load_smiles(compound_ids: list[int]) -> pd.DataFrame:
    eng = get_engine()
    ids_str = ",".join(str(i) for i in compound_ids)
    return pd.read_sql(
        f"SELECT id AS compound_id, std_smiles AS smiles FROM compounds "
        f"WHERE id IN ({ids_str}) AND std_smiles IS NOT NULL",
        eng,
    )


def main() -> None:
    print("=" * 70)
    print(" Pseudo-label distribution shift analysis")
    print("=" * 70)

    eng = get_engine()
    train_df = pd.read_sql(
        "SELECT t.compound_id, c.std_smiles AS smiles "
        "FROM train_activity t JOIN compounds c ON c.id = t.compound_id "
        "WHERE c.std_smiles IS NOT NULL ORDER BY t.compound_id",
        eng,
    )
    test_df = pd.read_sql(
        "SELECT t.compound_id, c.std_smiles AS smiles "
        "FROM test_activity t JOIN compounds c ON c.id = t.compound_id "
        "WHERE c.std_smiles IS NOT NULL ORDER BY t.compound_id",
        eng,
    )
    pseudo_ids = pd.read_parquet(PSEUDO_PATH)["compound_id"].astype(int).tolist()
    pseudo_df = (
        load_smiles(pseudo_ids).sort_values("compound_id").reset_index(drop=True)
    )

    print(f"  train:  {len(train_df)} compounds")
    print(f"  test:   {len(test_df)} compounds")
    print(f"  pseudo: {len(pseudo_df)} compounds")

    # Morgan fingerprints
    print("\n  Computing Morgan FPs...")
    train_fps = morgan_fps(train_df["smiles"].tolist())
    test_fps = morgan_fps(test_df["smiles"].tolist())
    pseudo_fps = morgan_fps(pseudo_df["smiles"].tolist())

    # UMAP fit on train only (same params as splits.umap_split_indices)
    print("  Fitting UMAP on train (n_components=10, jaccard, n_neighbors=30)...")
    import umap

    reducer = umap.UMAP(
        n_components=10, metric="jaccard", random_state=SEED, n_neighbors=30
    )
    train_emb = reducer.fit_transform(train_fps)
    print("  Transforming pseudo and test into the same space...")
    pseudo_emb = reducer.transform(pseudo_fps)
    test_emb = reducer.transform(test_fps)

    # UMAP transform with jaccard can emit NaN for novel points; drop them
    pseudo_valid = ~np.isnan(pseudo_emb).any(axis=1)
    test_valid = ~np.isnan(test_emb).any(axis=1)
    if (~pseudo_valid).sum() or (~test_valid).sum():
        print(
            f"  WARN: UMAP transform produced NaN for "
            f"{(~pseudo_valid).sum()} pseudo and {(~test_valid).sum()} test "
            f"compounds — dropping from cluster/distance analysis"
        )
    pseudo_emb = pseudo_emb[pseudo_valid]
    test_emb = test_emb[test_valid]

    # KMeans on train only (same as splits)
    print(f"  KMeans (n_clusters={N_CLUSTERS}) on train...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    train_clusters = km.fit_predict(train_emb)
    pseudo_clusters = km.predict(pseudo_emb)
    test_clusters = km.predict(test_emb)

    # ----- 1. Nearest-train-neighbor distance (Euclidean in UMAP space) -----
    print("\n=== 1. Nearest-train-neighbor distance (UMAP-10D Euclidean) ===")
    nn = NearestNeighbors(n_neighbors=1).fit(train_emb)
    d_train_self, _ = nn.kneighbors(train_emb, n_neighbors=2)
    d_train_self = d_train_self[:, 1]  # exclude self
    d_pseudo, _ = nn.kneighbors(pseudo_emb, n_neighbors=1)
    d_test, _ = nn.kneighbors(test_emb, n_neighbors=1)

    def summary(name: str, d: np.ndarray) -> None:
        q = np.quantile(d, [0.25, 0.5, 0.75, 0.95])
        print(
            f"  {name:8s} mean={d.mean():.3f} median={q[1]:.3f} "
            f"q25={q[0]:.3f} q75={q[2]:.3f} q95={q[3]:.3f}"
        )

    summary("train", d_train_self.flatten())
    summary("test", d_test.flatten())
    summary("pseudo", d_pseudo.flatten())

    # KS test pseudo vs test distances
    from scipy.stats import ks_2samp

    ks = ks_2samp(d_pseudo.flatten(), d_test.flatten())
    print(f"\n  KS(pseudo, test): D={ks.statistic:.3f}, p={ks.pvalue:.2e}")
    ks2 = ks_2samp(d_pseudo.flatten(), d_train_self.flatten())
    print(f"  KS(pseudo, train_self): D={ks2.statistic:.3f}, p={ks2.pvalue:.2e}")

    # ----- 2. Cluster occupancy -----
    print("\n=== 2. Cluster occupancy (50 train clusters) ===")
    train_counts = Counter(train_clusters)
    pseudo_counts = Counter(pseudo_clusters)
    test_counts = Counter(test_clusters)

    train_share = np.array([train_counts.get(i, 0) for i in range(N_CLUSTERS)]) / len(
        train_clusters
    )
    pseudo_share = np.array([pseudo_counts.get(i, 0) for i in range(N_CLUSTERS)]) / len(
        pseudo_clusters
    )
    test_share = np.array([test_counts.get(i, 0) for i in range(N_CLUSTERS)]) / len(
        test_clusters
    )

    print(
        f"  Empty-in-train clusters used by pseudo: "
        f"{sum(1 for i in range(N_CLUSTERS) if train_counts.get(i, 0) < 5 and pseudo_counts.get(i, 0) > 0)}"
        f" (cluster size <5 in train)"
    )

    # Cosine similarity of share vectors
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"  cos(train_share, pseudo_share) = {cos(train_share, pseudo_share):.3f}")
    print(f"  cos(train_share, test_share)   = {cos(train_share, test_share):.3f}")
    print(f"  cos(pseudo_share, test_share)  = {cos(pseudo_share, test_share):.3f}")

    # JS divergence (smoothed)
    def js(p, q):
        eps = 1e-9
        p = (p + eps) / (p + eps).sum()
        q = (q + eps) / (q + eps).sum()
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    print(f"  JS(train, pseudo) = {js(train_share, pseudo_share):.4f}")
    print(f"  JS(train, test)   = {js(train_share, test_share):.4f}")
    print(f"  JS(pseudo, test)  = {js(pseudo_share, test_share):.4f}")

    # Top-5 clusters where pseudo is most over-represented
    over = pseudo_share - train_share
    top5 = np.argsort(over)[::-1][:5]
    print("\n  Top 5 clusters where pseudo is over-represented vs train:")
    print("    cluster | train% | pseudo% | test%")
    for ci in top5:
        print(
            f"    {ci:>7d} | {train_share[ci] * 100:5.1f} | "
            f"{pseudo_share[ci] * 100:6.1f} | {test_share[ci] * 100:5.1f}"
        )

    # ----- 3. 2D UMAP for visualization -----
    print("\n=== 3. 2D UMAP scatter (visualization) ===")
    reducer2 = umap.UMAP(
        n_components=2, metric="jaccard", random_state=SEED, n_neighbors=30
    )
    train_2d = reducer2.fit_transform(train_fps)
    pseudo_2d = reducer2.transform(pseudo_fps)
    test_2d = reducer2.transform(test_fps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax in axes:
        ax.scatter(
            train_2d[:, 0],
            train_2d[:, 1],
            s=4,
            alpha=0.4,
            color="#888888",
            label="train",
        )
    axes[0].set_title("Train (4140)")
    axes[0].legend(loc="best")

    axes[1].scatter(
        pseudo_2d[:, 0],
        pseudo_2d[:, 1],
        s=4,
        alpha=0.5,
        color="#d62728",
        label="pseudo",
    )
    axes[1].set_title(f"Train + Pseudo ({len(pseudo_2d)})")
    axes[1].legend(loc="best")

    axes[2].scatter(
        test_2d[:, 0], test_2d[:, 1], s=10, alpha=0.7, color="#1f77b4", label="test"
    )
    axes[2].set_title(f"Train + Test ({len(test_2d)})")
    axes[2].legend(loc="best")

    fig.suptitle("UMAP 2D projection (Morgan r=2 / jaccard, fit on train)")
    fig.tight_layout()
    out = FIG_DIR.joinpath("pseudo_umap_distribution.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Distance histograms
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(d_pseudo.max(), d_test.max(), d_train_self.max()), 60)
    ax.hist(
        d_train_self,
        bins=bins,
        alpha=0.5,
        label=f"train→train (n={len(d_train_self)})",
        color="#888888",
    )
    ax.hist(
        d_test.flatten(),
        bins=bins,
        alpha=0.5,
        label=f"test→train (n={len(d_test)})",
        color="#1f77b4",
    )
    ax.hist(
        d_pseudo.flatten(),
        bins=bins,
        alpha=0.5,
        label=f"pseudo→train (n={len(d_pseudo)})",
        color="#d62728",
    )
    ax.set_xlabel("Distance to nearest train neighbor (UMAP-10D Euclidean)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Nearest-train-neighbor distance distributions")
    fig.tight_layout()
    out = FIG_DIR.joinpath("pseudo_nn_distance.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")

    print("\n" + "=" * 70)
    print(" Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
