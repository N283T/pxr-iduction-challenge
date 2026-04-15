#!/usr/bin/env -S pixi run python
"""Chemotype clustering of train + test with activity overlay.

Morgan FP -> UMAP -> HDBSCAN. The scatter is colored by train vs test so we
can see where train-only regions are, and the per-cluster summary reports
pEC50 stats + number of test neighbors. That summary is what actually
identifies "chemotypes the model cannot verify against test" -- size or
pose alone are not enough.

Outputs:
  - eda_redo_03_chemotype.png        - 2-panel: UMAP by split + UMAP by train pEC50
  - 03_chemotype_clusters.parquet    - per-cluster: N train/test, pEC50 stats, potent count
  - 03_chemotype_embedding.parquet   - per-compound umap coords + cluster id
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import HDBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from eda_redo import POTENT_PEC50, load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER_SIZE = 20


def _morgan_matrix(smiles: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    arr = np.zeros((len(smiles), MORGAN_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        row = np.zeros(MORGAN_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        arr[i] = row
    return arr


def main() -> None:
    df_all = load_master()
    df = df_all[df_all["split"].isin(["train", "test"])].copy()
    df = df.reset_index(drop=True)
    print(
        f"[03] compounds to embed: {len(df):,}"
        f"  (train {len(df[df['split'] == 'train']):,} + test {len(df[df['split'] == 'test']):,})"
    )

    # Prefer standardized SMILES to keep salts/tautomers consistent.
    smiles = df["std_smiles"].fillna(df["smiles"]).tolist()
    print(
        f"[03] computing Morgan fingerprints (r={MORGAN_RADIUS}, bits={MORGAN_BITS}) ..."
    )
    X = _morgan_matrix(smiles)

    print(f"[03] UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) ...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="jaccard",
        random_state=42,
    )
    emb = reducer.fit_transform(X)

    print(f"[03] HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}) ...")
    clusterer = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, n_jobs=-1)
    labels = clusterer.fit_predict(emb)
    n_clusters = int((labels >= 0).any()) and int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    print(f"[03] {n_clusters} clusters + {n_noise} noise points")

    df["umap_x"] = emb[:, 0]
    df["umap_y"] = emb[:, 1]
    df["cluster"] = labels

    # ------------------------------------------------------------------
    # Per-cluster summary
    # ------------------------------------------------------------------
    summary_rows = []
    for cid, sub in df.groupby("cluster"):
        tr_sub = sub[sub["split"] == "train"]
        te_sub = sub[sub["split"] == "test"]
        summary_rows.append(
            {
                "cluster": int(cid),
                "n_total": len(sub),
                "n_train": len(tr_sub),
                "n_test": len(te_sub),
                "frac_test": len(te_sub) / max(len(sub), 1),
                "pec50_median": tr_sub["train_pec50"].median(),
                "pec50_mean": tr_sub["train_pec50"].mean(),
                "pec50_max": tr_sub["train_pec50"].max(),
                "n_potent_train": int(tr_sub["train_is_potent"].fillna(False).sum()),
                "pose_dist_median": tr_sub["b2_pocket_distance_a"].median(),
                "pose_outliers_train": int((tr_sub["b2_pocket_distance_a"] > 4).sum()),
                "ha_median_train": tr_sub["num_heavy_atoms"].median(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("n_train", ascending=False)
    print()
    print("[03] cluster summary (top 15 by n_train):")
    cols_print = [
        "cluster",
        "n_total",
        "n_train",
        "n_test",
        "frac_test",
        "pec50_median",
        "n_potent_train",
        "pose_outliers_train",
        "ha_median_train",
    ]
    print(
        summary[cols_print]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    # Train-only clusters (never seen in test) ordered by pEC50 "loss"
    train_only = summary[(summary["n_test"] == 0) & (summary["cluster"] >= 0)].copy()
    train_only = train_only.sort_values("n_train", ascending=False)
    print()
    print(f"[03] train-only clusters (no test neighbors): {len(train_only)}")
    print(
        train_only[cols_print]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    # ------------------------------------------------------------------
    # Save parquets
    # ------------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    emb_keep = [
        "compound_id",
        "split",
        "smiles",
        "umap_x",
        "umap_y",
        "cluster",
        "num_heavy_atoms",
        "amw",
        "logp",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "train_is_potent",
        "b2_pocket_distance_a",
        "b2_confidence",
        "b2_affinity_pred",
    ]
    emb_path = DATA_DIR.joinpath("03_chemotype_embedding.parquet")
    df[emb_keep].to_parquet(emb_path, index=False)
    print(f"[03] wrote {emb_path}")

    sum_path = DATA_DIR.joinpath("03_chemotype_clusters.parquet")
    summary.to_parquet(sum_path, index=False)
    print(f"[03] wrote {sum_path}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: split coloring
    ax = axes[0]
    tr_mask = df["split"] == "train"
    te_mask = df["split"] == "test"
    ax.scatter(
        df.loc[tr_mask, "umap_x"],
        df.loc[tr_mask, "umap_y"],
        s=4,
        alpha=0.35,
        color="C0",
        label=f"train (N={tr_mask.sum()})",
    )
    ax.scatter(
        df.loc[te_mask, "umap_x"],
        df.loc[te_mask, "umap_y"],
        s=18,
        alpha=0.85,
        color="C1",
        marker="D",
        label=f"test (N={te_mask.sum()})",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Train vs test chemical space")
    ax.legend(loc="best")

    # Panel B: train pEC50 overlay (same coords); grey = no pEC50 (test), colored = train
    ax = axes[1]
    ax.scatter(
        df.loc[te_mask, "umap_x"],
        df.loc[te_mask, "umap_y"],
        s=18,
        alpha=0.5,
        color="lightgrey",
        marker="D",
        label="test (blind)",
    )
    tr_plot = df[tr_mask]
    sc = ax.scatter(
        tr_plot["umap_x"],
        tr_plot["umap_y"],
        c=tr_plot["train_pec50"],
        cmap="viridis",
        s=6,
        alpha=0.7,
        vmin=tr_plot["train_pec50"].quantile(0.02),
        vmax=tr_plot["train_pec50"].quantile(0.98),
    )
    pot = tr_plot[tr_plot["train_is_potent"].fillna(False)]
    ax.scatter(
        pot["umap_x"],
        pot["umap_y"],
        s=40,
        edgecolors="red",
        facecolors="none",
        linewidths=0.8,
        label=f"potent (pEC50 >= {POTENT_PEC50}, N={len(pot)})",
    )
    plt.colorbar(sc, ax=ax, label="train pEC50")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Activity overlay (test in grey, potent circled red)")
    ax.legend(loc="best")

    fig.suptitle("Issue #52 03: chemotype UMAP with activity overlay", fontsize=13)
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_03_chemotype.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] wrote {fig_path}")


if __name__ == "__main__":
    main()
