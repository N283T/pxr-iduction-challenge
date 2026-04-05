#!/usr/bin/env -S pixi run python
"""Chemical space EDA: train vs test similarity, clustering, and OOF error analysis.

Addresses issues #17 (OOF-LB gap) and #20 (ensemble optimization).
Outputs text summary + saves PNG figures to docs/figures/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import pdist, squareform

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

from data import get_conn, load_train_smiles_target, load_test_smiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_morgan_fps(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprints as RDKit ExplicitBitVect objects."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol))
    return fps


def compute_morgan_np(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprints as numpy array."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        arr.append(gen.GetFingerprintAsNumPy(mol))
    return np.array(arr, dtype=np.float32)


def nn_similarity(
    query_fps: list,
    ref_fps: list,
    k: int = 5,
) -> np.ndarray:
    """Compute mean Tanimoto similarity of each query to its k nearest neighbors in ref."""
    nn_sims = []
    for qfp in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)
        top_k = sorted(sims, reverse=True)[:k]
        nn_sims.append(np.mean(top_k))
    return np.array(nn_sims)


def butina_cluster(fps: list, cutoff: float = 0.4) -> list[tuple]:
    """Butina clustering based on Tanimoto distance."""
    n = len(fps)
    # Compute pairwise distances (upper triangle, 1 - Tanimoto)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return clusters


# ---------------------------------------------------------------------------
# 1. Train vs Test nearest-neighbor similarity (Sheridan metric)
# ---------------------------------------------------------------------------

def analyze_nn_similarity(train_df, test_df):
    """Compute and report Sheridan's NN similarity metric."""
    print("=" * 70)
    print("1. NEAREST NEIGHBOR SIMILARITY (Sheridan metric)")
    print("=" * 70)

    train_fps = compute_morgan_fps(train_df["smiles"].tolist())
    test_fps = compute_morgan_fps(test_df["smiles"].tolist())

    # Test -> Train similarity (how similar is each test compound to training set?)
    test_to_train = nn_similarity(test_fps, train_fps, k=5)

    # Train -> Train (leave-one-out style, for comparison)
    # For each train compound, compute NN sim to rest of train
    train_to_train = []
    for i, fp in enumerate(train_fps):
        ref = train_fps[:i] + train_fps[i + 1 :]
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref)
        top_k = sorted(sims, reverse=True)[:5]
        train_to_train.append(np.mean(top_k))
    train_to_train = np.array(train_to_train)

    print(f"\nTest->Train 5-NN similarity:")
    print(f"  Mean: {test_to_train.mean():.4f}  Std: {test_to_train.std():.4f}")
    print(f"  Min:  {test_to_train.min():.4f}  Max: {test_to_train.max():.4f}")
    print(f"  Q25:  {np.percentile(test_to_train, 25):.4f}  "
          f"Median: {np.median(test_to_train):.4f}  "
          f"Q75: {np.percentile(test_to_train, 75):.4f}")

    print(f"\nTrain->Train 5-NN similarity (baseline):")
    print(f"  Mean: {train_to_train.mean():.4f}  Std: {train_to_train.std():.4f}")
    print(f"  Min:  {train_to_train.min():.4f}  Max: {train_to_train.max():.4f}")
    print(f"  Q25:  {np.percentile(train_to_train, 25):.4f}  "
          f"Median: {np.median(train_to_train):.4f}  "
          f"Q75: {np.percentile(train_to_train, 75):.4f}")

    gap = train_to_train.mean() - test_to_train.mean()
    print(f"\n  Similarity gap (train-train minus test-train): {gap:.4f}")
    if gap > 0.05:
        print("  ⚠ Test compounds are notably less similar to train than train is to itself.")
        print("  This suggests chemical space shift that CV may not capture.")
    else:
        print("  Test and train occupy similar chemical space.")

    # Fraction of test compounds with low similarity
    thresholds = [0.3, 0.4, 0.5]
    print(f"\n  Test compounds with low NN similarity to train:")
    for th in thresholds:
        frac = (test_to_train < th).mean()
        print(f"    < {th}: {frac:.1%} ({(test_to_train < th).sum()}/{len(test_to_train)})")

    return train_to_train, test_to_train


# ---------------------------------------------------------------------------
# 2. Butina clustering analysis
# ---------------------------------------------------------------------------

def analyze_butina(train_df, test_df):
    """Butina clustering of train+test and cluster distribution analysis."""
    print("\n" + "=" * 70)
    print("2. BUTINA CLUSTERING ANALYSIS")
    print("=" * 70)

    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    n_train = len(train_df)
    n_test = len(test_df)
    labels = ["train"] * n_train + ["test"] * n_test

    all_fps = compute_morgan_fps(all_smiles)

    for cutoff in [0.3, 0.4, 0.5]:
        clusters = butina_cluster(all_fps, cutoff=cutoff)
        n_clusters = len(clusters)
        singleton_count = sum(1 for c in clusters if len(c) == 1)

        # Analyze train/test distribution per cluster
        test_only_clusters = 0
        train_only_clusters = 0
        mixed_clusters = 0

        cluster_sizes = []
        for c in clusters:
            c_labels = [labels[i] for i in c]
            has_train = "train" in c_labels
            has_test = "test" in c_labels
            if has_train and has_test:
                mixed_clusters += 1
            elif has_test:
                test_only_clusters += 1
            else:
                train_only_clusters += 1
            cluster_sizes.append(len(c))

        # Test compounds in test-only clusters
        test_in_test_only = sum(
            sum(1 for i in c if labels[i] == "test")
            for c in clusters
            if all(labels[i] == "test" for i in c)
        )

        print(f"\n  Cutoff={cutoff} (Tanimoto distance):")
        print(f"    Total clusters: {n_clusters}  "
              f"(singletons: {singleton_count}, {singleton_count / n_clusters:.0%})")
        print(f"    Cluster sizes: min={min(cluster_sizes)} median={int(np.median(cluster_sizes))} "
              f"max={max(cluster_sizes)} mean={np.mean(cluster_sizes):.1f}")
        print(f"    Train-only:  {train_only_clusters}")
        print(f"    Test-only:   {test_only_clusters}  "
              f"({test_in_test_only} test compounds, {test_in_test_only / n_test:.1%})")
        print(f"    Mixed:       {mixed_clusters}")

    return clusters


# ---------------------------------------------------------------------------
# 3. Murcko scaffold analysis (more detailed)
# ---------------------------------------------------------------------------

def analyze_scaffolds(train_df, test_df):
    """Detailed Murcko scaffold analysis."""
    print("\n" + "=" * 70)
    print("3. MURCKO SCAFFOLD ANALYSIS")
    print("=" * 70)

    from rdkit.Chem.Scaffolds import MurckoScaffold

    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            return ""

    train_scaffolds = [get_scaffold(s) for s in train_df["smiles"]]
    test_scaffolds = [get_scaffold(s) for s in test_df["smiles"]]

    train_unique = set(train_scaffolds)
    test_unique = set(test_scaffolds)
    overlap = train_unique & test_unique

    # Scaffold frequency in train
    from collections import Counter
    train_counts = Counter(train_scaffolds)
    train_singletons = sum(1 for v in train_counts.values() if v == 1)

    test_counts = Counter(test_scaffolds)
    test_singletons = sum(1 for v in test_counts.values() if v == 1)

    print(f"\n  Train: {len(train_unique)} unique scaffolds out of {len(train_df)} compounds")
    print(f"    Singletons: {train_singletons} ({train_singletons / len(train_unique):.1%})")
    print(f"    Top-5 scaffolds: {train_counts.most_common(5)}")

    print(f"\n  Test: {len(test_unique)} unique scaffolds out of {len(test_df)} compounds")
    print(f"    Singletons: {test_singletons} ({test_singletons / len(test_unique):.1%})")

    print(f"\n  Scaffold overlap (train ∩ test): {len(overlap)} scaffolds")
    # How many test compounds have a scaffold seen in train?
    test_with_train_scaffold = sum(1 for s in test_scaffolds if s in train_unique)
    print(f"    Test compounds with train scaffold: {test_with_train_scaffold}/{len(test_df)} "
          f"({test_with_train_scaffold / len(test_df):.1%})")
    test_novel = len(test_df) - test_with_train_scaffold
    print(f"    Test compounds with novel scaffold: {test_novel}/{len(test_df)} "
          f"({test_novel / len(test_df):.1%})")

    return train_scaffolds, test_scaffolds


# ---------------------------------------------------------------------------
# 4. UMAP visualization
# ---------------------------------------------------------------------------

def analyze_umap(train_df, test_df):
    """UMAP projection of train+test chemical space."""
    print("\n" + "=" * 70)
    print("4. UMAP CHEMICAL SPACE PROJECTION")
    print("=" * 70)

    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  UMAP or matplotlib not available, skipping visualization.")
        return None

    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    fps_np = compute_morgan_np(all_smiles)
    n_train = len(train_df)

    reducer = umap.UMAP(n_components=2, metric="jaccard", random_state=42, n_neighbors=30)
    embedding = reducer.fit_transform(fps_np)

    train_emb = embedding[:n_train]
    test_emb = embedding[n_train:]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: train vs test
    ax = axes[0]
    ax.scatter(train_emb[:, 0], train_emb[:, 1], c="steelblue", alpha=0.3, s=8, label="Train")
    ax.scatter(test_emb[:, 0], test_emb[:, 1], c="crimson", alpha=0.6, s=15, label="Test")
    ax.set_title("Train vs Test Chemical Space (UMAP)")
    ax.legend()
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Right: train colored by pEC50
    ax = axes[1]
    sc = ax.scatter(
        train_emb[:, 0], train_emb[:, 1],
        c=train_df["pec50"].values, cmap="viridis", alpha=0.5, s=8,
    )
    ax.scatter(test_emb[:, 0], test_emb[:, 1], c="crimson", alpha=0.3, s=10, marker="x", label="Test")
    ax.set_title("Train pEC50 in Chemical Space (UMAP)")
    ax.legend()
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(sc, ax=ax, label="pEC50")

    fig_dir = Path(__file__).resolve().parent.parent.parent.joinpath("docs", "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir.joinpath("umap_train_test.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved UMAP plot to {fig_path}")
    plt.close(fig)

    return embedding


# ---------------------------------------------------------------------------
# 5. OOF error analysis
# ---------------------------------------------------------------------------

def analyze_oof_errors(train_df, train_to_train_sim):
    """Analyze OOF prediction errors vs chemical space properties."""
    print("\n" + "=" * 70)
    print("5. OOF ERROR ANALYSIS")
    print("=" * 70)

    # Load OOF predictions from DB (best single model and ensemble)
    conn = get_conn()
    cur = conn.cursor()

    # Get experiments with OOF predictions
    cur.execute("""
        SELECT e.id, e.name, e.feature_set,
               (SELECT AVG(rae) FROM experiment_cv_results WHERE experiment_id = e.id) as rae_mean
        FROM experiments e
        WHERE EXISTS (SELECT 1 FROM experiment_oof_predictions WHERE experiment_id = e.id)
        ORDER BY rae_mean
    """)
    experiments = cur.fetchall()
    cur.close()
    conn.close()

    if not experiments:
        print("  No OOF predictions found in DB. Skipping.")
        return

    print(f"\n  Experiments with OOF predictions: {len(experiments)}")
    for exp_id, name, feat, rae in experiments[:5]:
        print(f"    [{exp_id}] {name} (RAE={rae:.4f})")

    # Use best single model for error analysis
    best_id = experiments[0][0]
    best_name = experiments[0][1]
    print(f"\n  Analyzing OOF errors for: {best_name} (id={best_id})")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT train_idx, oof_prediction FROM experiment_oof_predictions "
        "WHERE experiment_id = %s ORDER BY train_idx",
        (best_id,),
    )
    oof_rows = cur.fetchall()
    cur.close()
    conn.close()

    if not oof_rows:
        print("  No OOF predictions found. Skipping.")
        return

    oof_preds = np.array([r[1] for r in oof_rows])
    y_true = train_df["pec50"].values

    if len(oof_preds) != len(y_true):
        print(f"  ⚠ OOF length mismatch: {len(oof_preds)} vs {len(y_true)}. Skipping.")
        return

    residuals = np.abs(y_true - oof_preds)

    # Correlation between NN similarity and OOF error
    from scipy import stats
    r, p = stats.spearmanr(train_to_train_sim, residuals)
    print(f"\n  Correlation (NN similarity vs |OOF error|):")
    print(f"    Spearman r = {r:.4f}  (p = {p:.2e})")
    if r < -0.1 and p < 0.01:
        print("    ⚠ Compounds in sparse regions (low NN sim) have larger errors.")
    elif r > 0.1 and p < 0.01:
        print("    Compounds in dense regions have larger errors (possible overfitting).")
    else:
        print("    No strong relationship between isolation and prediction error.")

    # Error by pEC50 range
    print(f"\n  OOF error by pEC50 range:")
    bins = [(1.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0)]
    for lo, hi in bins:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        bin_mae = residuals[mask].mean()
        bin_rae = residuals[mask].sum() / np.abs(y_true[mask] - y_true.mean()).sum()
        print(f"    pEC50 [{lo:.0f}, {hi:.0f}): n={mask.sum():4d}  "
              f"MAE={bin_mae:.3f}  RAE={bin_rae:.3f}")

    # Top 20 worst predictions
    worst_idx = np.argsort(residuals)[-20:][::-1]
    print(f"\n  Top 20 worst OOF predictions:")
    print(f"    {'idx':>5} {'true':>6} {'pred':>6} {'|err|':>6} {'NN_sim':>7}")
    for idx in worst_idx:
        print(f"    {idx:5d} {y_true[idx]:6.2f} {oof_preds[idx]:6.2f} "
              f"{residuals[idx]:6.2f} {train_to_train_sim[idx]:7.4f}")

    # Error vs NN similarity bins
    print(f"\n  OOF error by NN similarity quartile:")
    quartiles = np.percentile(train_to_train_sim, [25, 50, 75])
    q_labels = [
        (0.0, quartiles[0], "Q1 (most isolated)"),
        (quartiles[0], quartiles[1], "Q2"),
        (quartiles[1], quartiles[2], "Q3"),
        (quartiles[2], 1.0, "Q4 (most connected)"),
    ]
    for lo, hi, label in q_labels:
        mask = (train_to_train_sim >= lo) & (train_to_train_sim < hi)
        if mask.sum() == 0:
            continue
        print(f"    {label:25s}: n={mask.sum():4d}  "
              f"MAE={residuals[mask].mean():.3f}  "
              f"median|err|={np.median(residuals[mask]):.3f}")

    # Plot: error vs similarity
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(train_to_train_sim, residuals, alpha=0.2, s=5, c="steelblue")
        ax.set_xlabel("5-NN Tanimoto Similarity (train-train)")
        ax.set_ylabel("|OOF Prediction Error|")
        ax.set_title(f"OOF Error vs NN Similarity\n(Spearman r={r:.3f})")

        ax = axes[1]
        ax.scatter(y_true, oof_preds, alpha=0.2, s=5, c="steelblue")
        lims = [min(y_true.min(), oof_preds.min()), max(y_true.max(), oof_preds.max())]
        ax.plot(lims, lims, "r--", alpha=0.5)
        ax.set_xlabel("True pEC50")
        ax.set_ylabel("OOF Predicted pEC50")
        ax.set_title(f"OOF: True vs Predicted ({best_name})")

        fig_dir = Path(__file__).resolve().parent.parent.parent.joinpath("docs", "figures")
        fig_path = fig_dir.joinpath("oof_error_analysis.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved OOF error plot to {fig_path}")
        plt.close(fig)
    except ImportError:
        pass

    return residuals


# ---------------------------------------------------------------------------
# 6. CV split comparison preview
# ---------------------------------------------------------------------------

def analyze_cv_splits(train_df):
    """Compare Murcko scaffold vs Butina cluster group counts for CV."""
    print("\n" + "=" * 70)
    print("6. CV SPLIT STRATEGY COMPARISON (group counts)")
    print("=" * 70)

    from splits import scaffold_split_indices, get_murcko_scaffold
    from collections import Counter

    smiles_list = train_df["smiles"].tolist()
    n = len(smiles_list)

    # Murcko scaffolds
    scaffolds = [get_murcko_scaffold(s) for s in smiles_list]
    scaffold_counts = Counter(scaffolds)
    scaffold_singletons = sum(1 for v in scaffold_counts.values() if v == 1)
    n_scaffolds = len(scaffold_counts)

    print(f"\n  Murcko scaffold groups: {n_scaffolds}")
    print(f"    Singletons: {scaffold_singletons} ({scaffold_singletons / n_scaffolds:.1%})")
    print(f"    Effective group ratio: {n_scaffolds / n:.2%} unique")

    # Butina clusters
    fps = compute_morgan_fps(smiles_list)
    for cutoff in [0.3, 0.4, 0.5, 0.6]:
        clusters = butina_cluster(fps, cutoff=cutoff)
        n_clusters = len(clusters)
        singleton_clusters = sum(1 for c in clusters if len(c) == 1)
        sizes = [len(c) for c in clusters]
        print(f"\n  Butina (cutoff={cutoff}): {n_clusters} clusters")
        print(f"    Singletons: {singleton_clusters} ({singleton_clusters / n_clusters:.1%})")
        print(f"    Sizes: min={min(sizes)} median={int(np.median(sizes))} "
              f"max={max(sizes)} mean={np.mean(sizes):.1f}")
        print(f"    Effective group ratio: {n_clusters / n:.2%} unique")

    # Key question: with 85%+ singletons, scaffold split ≈ random split
    print(f"\n  ⚠ Key insight:")
    print(f"    If >80% of groups are singletons, the split degenerates toward random.")
    print(f"    Butina with lower cutoff produces fewer, larger clusters → stricter split.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data from DB...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    print(f"  Train: {len(train_df)} compounds, Test: {len(test_df)} compounds\n")

    # 1. NN similarity
    train_to_train_sim, test_to_train_sim = analyze_nn_similarity(train_df, test_df)

    # 2. Butina clustering
    analyze_butina(train_df, test_df)

    # 3. Scaffold analysis
    analyze_scaffolds(train_df, test_df)

    # 4. UMAP
    analyze_umap(train_df, test_df)

    # 5. OOF errors
    analyze_oof_errors(train_df, train_to_train_sim)

    # 6. CV split comparison
    analyze_cv_splits(train_df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
