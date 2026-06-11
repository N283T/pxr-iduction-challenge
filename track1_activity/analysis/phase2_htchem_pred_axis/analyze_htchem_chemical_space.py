#!/usr/bin/env -S pixi run python
"""Analyze HTChem chemical-space relationship to Track 1 train/test."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))

from data import get_engine  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "chemical_space"
ASSET_DIR = (
    REPO_ROOT / "docs" / "track1_explain" / "assets" / "phase2_htchem_chemical_space"
)
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_htchem_chemical_space.md"


def load_train() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS row_id,
            t.compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            t.pec50
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_test() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS row_id,
            t.compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS as1_pec50
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_htchem() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            h.compound_id,
            c.molecule_name,
            COALESCE(c.std_smiles, c.smiles) AS smiles,
            h.source_type,
            h.corrected_pec50,
            h.corrected_pec50_se,
            h.product_yield_percent
        FROM htchem_activity h
        JOIN compounds c ON c.id = h.compound_id
        WHERE h.corrected_pec50 IS NOT NULL
        ORDER BY h.compound_id
        """,
        get_engine(),
    )


def tanimoto_matrix(query_fp: np.ndarray, ref_fp: np.ndarray) -> np.ndarray:
    query = query_fp.astype(bool)
    ref = ref_fp.astype(bool)
    inter = query.astype(np.uint16) @ ref.astype(np.uint16).T
    union = query.sum(axis=1, keepdims=True) + ref.sum(axis=1, keepdims=True).T - inter
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def nn_summary(
    query_name: str,
    ref_name: str,
    sim: np.ndarray,
    query_meta: pd.DataFrame,
    ref_meta: pd.DataFrame,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    order = np.argsort(-sim, axis=1)
    top1_idx = order[:, 0]
    top5_idx = order[:, : min(5, sim.shape[1])]
    top1 = sim[np.arange(sim.shape[0]), top1_idx]
    top5_mean = np.take_along_axis(sim, top5_idx, axis=1).mean(axis=1)
    row = {
        "query": query_name,
        "ref": ref_name,
        "n_query": int(sim.shape[0]),
        "n_ref": int(sim.shape[1]),
        "top1_mean": float(top1.mean()),
        "top1_median": float(np.median(top1)),
        "top1_p90": float(np.quantile(top1, 0.90)),
        "top1_max": float(top1.max()),
        "top5_mean": float(top5_mean.mean()),
    }
    for threshold in (0.30, 0.40, 0.50, 0.60):
        row[f"top1_ge_{threshold:.2f}"] = int((top1 >= threshold).sum())

    nearest = query_meta.copy()
    nearest[f"nn_{ref_name}_compound_id"] = ref_meta["compound_id"].to_numpy()[top1_idx]
    nearest[f"nn_{ref_name}_molecule_name"] = ref_meta["molecule_name"].to_numpy()[
        top1_idx
    ]
    nearest[f"nn_{ref_name}_tanimoto"] = top1
    if "pec50" in ref_meta.columns:
        nearest[f"nn_{ref_name}_pec50"] = ref_meta["pec50"].to_numpy()[top1_idx]
    if "corrected_pec50" in ref_meta.columns:
        nearest[f"nn_{ref_name}_corrected_pec50"] = ref_meta[
            "corrected_pec50"
        ].to_numpy()[top1_idx]
    return row, nearest


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_summary(
    train: pd.DataFrame, test: pd.DataFrame, htchem: pd.DataFrame
) -> pd.DataFrame:
    frames = {
        "train": train,
        "test": test,
        "as1": test[test["split"].eq("AS1")],
        "as2": test[test["split"].eq("AS2")],
        "htchem": htchem,
    }
    scaffold_sets = {
        name: set(frame["smiles"].map(scaffold)) - {""}
        for name, frame in frames.items()
    }
    rows = []
    for name, frame in frames.items():
        scaffolds = frame["smiles"].map(scaffold)
        unique = set(scaffolds) - {""}
        rows.append(
            {
                "set": name,
                "n_compounds": int(len(frame)),
                "n_scaffolds": int(len(unique)),
                "overlap_train_scaffolds": int(len(unique & scaffold_sets["train"])),
                "overlap_test_scaffolds": int(len(unique & scaffold_sets["test"])),
                "overlap_htchem_scaffolds": int(len(unique & scaffold_sets["htchem"])),
                "compounds_with_train_scaffold": int(
                    scaffolds.isin(scaffold_sets["train"]).sum()
                ),
                "compounds_with_htchem_scaffold": int(
                    scaffolds.isin(scaffold_sets["htchem"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_umap(df: pd.DataFrame) -> None:
    colors = {
        "train": "#707070",
        "AS1": "#1f77b4",
        "AS2": "#ff7f0e",
        "htchem_crude": "#2ca02c",
        "htchem_semi_pure": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for split, sub in df.groupby("space_label", sort=False):
        size = 8 if split == "train" else 18
        alpha = 0.28 if split == "train" else 0.78
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=size,
            alpha=alpha,
            c=colors.get(split, "#9467bd"),
            label=f"{split} ({len(sub)})",
            linewidths=0,
        )
    ax.set_xlabel("Morgan UMAP 1")
    ax.set_ylabel("Morgan UMAP 2")
    ax.set_title("HTChem vs Track 1 train/test chemical space")
    ax.legend(markerscale=1.8, fontsize=8, frameon=False)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ASSET_DIR / "htchem_train_test_morgan_umap.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def write_doc(
    nn: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    htchem_cluster_examples: pd.DataFrame,
) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    rel_img = "assets/phase2_htchem_chemical_space/htchem_train_test_morgan_umap.png"
    text = f"""# Phase 2 HTChem Chemical Space

Purpose: check whether HTChem sits near Track 1 train/test compounds or mostly occupies a separate local SAR island.

![HTChem train/test Morgan UMAP]({rel_img})

## Morgan Nearest-Neighbor Summary

{nn.to_markdown(index=False, floatfmt=".4f")}

## Scaffold Overlap

{scaffold_df.to_markdown(index=False)}

## UMAP/KMeans Cluster Overlap

{cluster_summary.to_markdown(index=False)}

## HTChem-Test Mixed Cluster Examples

{htchem_cluster_examples.to_markdown(index=False, floatfmt=".4f")}

## Read

HTChem has almost no row overlap with Track 1, so this checks chemistry rather than IDs. Use the AS1/AS2-to-HTChem nearest-neighbor counts to decide whether HTChem is likely to help a meaningful blind subset or only a small local region.
"""
    DOC_PATH.write_text(text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    test = load_test()
    test["split"] = np.where(test["as1_pec50"].notna(), "AS1", "AS2")
    htchem = load_htchem()

    train_fp = _morgan_fp_matrix(train["smiles"].tolist())
    test_fp = _morgan_fp_matrix(test["smiles"].tolist())
    as1 = test[test["split"].eq("AS1")].copy()
    as2 = test[test["split"].eq("AS2")].copy()
    as1_fp = _morgan_fp_matrix(as1["smiles"].tolist())
    as2_fp = _morgan_fp_matrix(as2["smiles"].tolist())
    htchem_fp = _morgan_fp_matrix(htchem["smiles"].tolist())

    nn_rows = []
    nearest_outputs = {}
    pairs = [
        ("htchem", "train", htchem_fp, train_fp, htchem, train),
        ("htchem", "test", htchem_fp, test_fp, htchem, test),
        ("htchem", "AS1", htchem_fp, as1_fp, htchem, as1),
        ("htchem", "AS2", htchem_fp, as2_fp, htchem, as2),
        ("AS1", "htchem", as1_fp, htchem_fp, as1, htchem),
        ("AS2", "htchem", as2_fp, htchem_fp, as2, htchem),
        ("test", "htchem", test_fp, htchem_fp, test, htchem),
    ]
    for query_name, ref_name, query_fp, ref_fp, query_meta, ref_meta in pairs:
        row, nearest = nn_summary(
            query_name,
            ref_name,
            tanimoto_matrix(query_fp, ref_fp),
            query_meta.reset_index(drop=True),
            ref_meta.reset_index(drop=True),
        )
        nn_rows.append(row)
        nearest_outputs[f"{query_name}_to_{ref_name}_nearest.csv"] = nearest

    nn = pd.DataFrame(nn_rows)
    nn.to_csv(OUT_DIR / "nn_summary.csv", index=False)
    for filename, frame in nearest_outputs.items():
        frame.to_csv(OUT_DIR / filename, index=False)

    scaffold_df = scaffold_summary(train, test, htchem)
    scaffold_df.to_csv(OUT_DIR / "scaffold_summary.csv", index=False)

    import umap

    combined = pd.concat(
        [
            train.assign(space_label="train")[
                ["compound_id", "molecule_name", "smiles", "space_label"]
            ],
            test.assign(space_label=test["split"])[
                ["compound_id", "molecule_name", "smiles", "space_label"]
            ],
            htchem.assign(space_label="htchem_" + htchem["source_type"])[
                ["compound_id", "molecule_name", "smiles", "space_label"]
            ],
        ],
        ignore_index=True,
    )
    combined_fp = _morgan_fp_matrix(combined["smiles"].tolist())
    reducer = umap.UMAP(
        n_components=2,
        metric="jaccard",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
    )
    coords = reducer.fit_transform(combined_fp.astype(bool))
    combined["umap_x"] = coords[:, 0]
    combined["umap_y"] = coords[:, 1]
    km = KMeans(n_clusters=50, random_state=42, n_init=10)
    combined["cluster"] = km.fit_predict(coords)
    combined.to_csv(OUT_DIR / "morgan_umap_coordinates.csv", index=False)
    plot_umap(combined)

    cluster_summary = (
        combined.pivot_table(
            index="cluster",
            columns="space_label",
            values="compound_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["train", "AS1", "AS2", "htchem_crude", "htchem_semi_pure"]:
        if col not in cluster_summary:
            cluster_summary[col] = 0
    cluster_summary["htchem_total"] = (
        cluster_summary["htchem_crude"] + cluster_summary["htchem_semi_pure"]
    )
    cluster_summary["test_total"] = cluster_summary["AS1"] + cluster_summary["AS2"]
    cluster_summary["has_htchem_and_test"] = (cluster_summary["htchem_total"] > 0) & (
        cluster_summary["test_total"] > 0
    )
    cluster_summary["has_htchem_and_train"] = (cluster_summary["htchem_total"] > 0) & (
        cluster_summary["train"] > 0
    )
    cluster_summary = cluster_summary.sort_values(
        ["htchem_total", "test_total"], ascending=False
    )
    cluster_summary.to_csv(OUT_DIR / "umap_kmeans_cluster_summary.csv", index=False)

    mixed_clusters = cluster_summary[cluster_summary["has_htchem_and_test"]].head(12)[
        "cluster"
    ]
    examples = combined[combined["cluster"].isin(mixed_clusters)].copy()
    examples = examples.sort_values(["cluster", "space_label", "molecule_name"])
    examples.groupby("cluster").head(12).to_csv(
        OUT_DIR / "htchem_test_mixed_cluster_examples.csv", index=False
    )

    write_doc(
        nn,
        cluster_summary.head(20),
        scaffold_df,
        examples.groupby("cluster").head(8)[
            [
                "cluster",
                "space_label",
                "molecule_name",
                "compound_id",
                "umap_x",
                "umap_y",
            ]
        ],
    )
    print(f"Wrote outputs to {OUT_DIR}")
    print(f"Wrote doc to {DOC_PATH}")
    print("\nNN summary:")
    print(nn.to_string(index=False))
    print("\nCluster overlap headline:")
    headline_cols = [
        "cluster",
        "train",
        "AS1",
        "AS2",
        "htchem_crude",
        "htchem_semi_pure",
        "htchem_total",
        "test_total",
    ]
    print(cluster_summary[headline_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
