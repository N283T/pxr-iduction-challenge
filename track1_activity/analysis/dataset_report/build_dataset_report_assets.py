#!/usr/bin/env python
"""Build figures used by the Track 1 dataset/split explanation report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdCoordGen
from rdkit.Chem import rdFingerprintGenerator


REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_PATH = REPO_ROOT / "data" / "eda_redo" / "master.parquet"
ASSET_DIR = REPO_ROOT / "docs" / "track1_explain" / "assets" / "dataset"
OUT_DIR = REPO_ROOT / "data" / "track1_explain" / "dataset_report"

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
RNG_SEED = 42

COLORS = {
    "train": "#3568a9",
    "test": "#e6862f",
    "single_only": "#4ca66a",
    "no_aux": "#c8c8c8",
    "single": "#6cbf86",
    "counter": "#9b7bb8",
}


def max_ring_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return 0
    return max((len(ring) for ring in mol.GetRingInfo().AtomRings()), default=0)


def load_master() -> pd.DataFrame:
    df = pd.read_parquet(MASTER_PATH).copy()
    df["plot_role"] = np.select(
        [df["in_test"], df["in_train"], df["in_single"]],
        ["test", "train", "single_only"],
        default="other",
    )
    df["display_smiles"] = df["std_smiles"].fillna(df["smiles"])
    return df


def morgan_matrix(smiles: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
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


def compute_or_load_umap(df: pd.DataFrame) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = OUT_DIR / "all_compound_morgan_umap.parquet"
    if emb_path.exists():
        return pd.read_parquet(emb_path)

    smiles = df["std_smiles"].fillna(df["smiles"]).tolist()
    x = morgan_matrix(smiles)
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="jaccard",
        random_state=RNG_SEED,
    )
    emb = reducer.fit_transform(x)
    keep = df[
        [
            "compound_id",
            "plot_role",
            "in_train",
            "in_test",
            "in_single",
            "in_counter",
            "train_pec50",
            "single_max_log2_fc",
            "counter_pec50",
        ]
    ].copy()
    keep["umap_x"] = emb[:, 0]
    keep["umap_y"] = emb[:, 1]
    keep.to_parquet(emb_path, index=False)
    return keep


def savefig(fig: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_compound_umap(emb: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    order = ["single_only", "train", "test"]
    labels = {
        "single_only": "single-conc only",
        "train": "train pEC50",
        "test": "blind test",
    }
    sizes = {"single_only": 5, "train": 6, "test": 18}
    alphas = {"single_only": 0.35, "train": 0.45, "test": 0.9}
    for role in order:
        sub = emb[emb["plot_role"] == role]
        axes[0].scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=sizes[role],
            alpha=alphas[role],
            color=COLORS[role],
            label=f"{labels[role]} (N={len(sub):,})",
            linewidths=0,
        )
    axes[0].set_title("Compound space by data role")
    axes[0].legend(loc="best", frameon=True)

    axes[1].scatter(
        emb["umap_x"],
        emb["umap_y"],
        s=3,
        alpha=0.16,
        color=COLORS["no_aux"],
        label="all compounds",
        linewidths=0,
    )
    single = emb[emb["in_single"]]
    axes[1].scatter(
        single["umap_x"],
        single["umap_y"],
        s=5,
        alpha=0.35,
        color=COLORS["single"],
        label=f"single-conc measured (N={len(single):,})",
        linewidths=0,
    )
    counter = emb[emb["in_counter"]]
    axes[1].scatter(
        counter["umap_x"],
        counter["umap_y"],
        s=12,
        alpha=0.75,
        facecolors="none",
        edgecolors=COLORS["counter"],
        linewidths=0.5,
        label=f"counter assay (N={len(counter):,})",
    )
    axes[1].set_title("Auxiliary assay coverage")
    axes[1].legend(loc="best", frameon=True)

    axes[2].scatter(
        emb["umap_x"],
        emb["umap_y"],
        s=3,
        alpha=0.12,
        color="lightgrey",
        linewidths=0,
    )
    tr = emb[emb["in_train"] & emb["train_pec50"].notna()]
    sc = axes[2].scatter(
        tr["umap_x"],
        tr["umap_y"],
        c=tr["train_pec50"],
        cmap="viridis",
        s=7,
        alpha=0.72,
        vmin=tr["train_pec50"].quantile(0.02),
        vmax=tr["train_pec50"].quantile(0.98),
        linewidths=0,
    )
    te = emb[emb["in_test"]]
    axes[2].scatter(
        te["umap_x"],
        te["umap_y"],
        s=18,
        alpha=0.85,
        color=COLORS["test"],
        marker="D",
        label="blind test",
        linewidths=0,
    )
    fig.colorbar(sc, ax=axes[2], shrink=0.86, label="train pEC50")
    axes[2].set_title("Train activity and blind test location")
    axes[2].legend(loc="best", frameon=True)

    for ax in axes:
        ax.set_xlabel("Morgan UMAP 1")
        ax.set_ylabel("Morgan UMAP 2")
        ax.grid(alpha=0.18, linewidth=0.5)

    savefig(fig, "all_compound_morgan_umap.png")


def plot_auxiliary_data(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    ax = axes[0, 0]

    coverage = pd.Series(
        {
            "train total": int(df["in_train"].sum()),
            "train + single": int((df["in_train"] & df["in_single"]).sum()),
            "train + counter": int((df["in_train"] & df["in_counter"]).sum()),
            "single-only aux": int(
                ((~df["in_train"]) & (~df["in_test"]) & df["in_single"]).sum()
            ),
            "blind test": int(df["in_test"].sum()),
        }
    )
    colors = ["#3568a9", "#6cbf86", "#9b7bb8", "#4ca66a", "#e6862f"]
    bars = ax.barh(coverage.index, coverage.values, color=colors, alpha=0.9)
    ax.bar_label(bars, labels=[f"{v:,}" for v in coverage.values], padding=4)
    ax.set_title("Local assay coverage")
    ax.set_xlabel("compounds")
    ax.invert_yaxis()
    ax.set_xlim(0, coverage.max() * 1.18)

    ax = axes[0, 1]
    bins = np.linspace(
        df["single_max_log2_fc"].quantile(0.005),
        df["single_max_log2_fc"].quantile(0.995),
        45,
    )
    for mask, label, color in [
        (df["split"].eq("single_only"), "single-only aux", COLORS["single_only"]),
        (df["in_train"] & df["in_single"], "train with single-conc", COLORS["train"]),
    ]:
        vals = df.loc[mask, "single_max_log2_fc"].dropna()
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{label} (N={len(vals):,})",
        )
    ax.set_title("Single-concentration response")
    ax.set_xlabel("max log2_fc per compound")
    ax.set_ylabel("density")
    ax.legend(frameon=True)

    ax = axes[1, 0]
    sub = df[
        df["in_train"] & df["single_max_log2_fc"].notna() & df["train_pec50"].notna()
    ]
    pearson = (
        sub[["single_max_log2_fc", "train_pec50"]].corr(method="pearson").iloc[0, 1]
    )
    spearman = (
        sub[["single_max_log2_fc", "train_pec50"]].corr(method="spearman").iloc[0, 1]
    )
    ax.scatter(
        sub["single_max_log2_fc"],
        sub["train_pec50"],
        s=9,
        alpha=0.35,
        color="#3568a9",
        linewidths=0,
    )
    ax.set_title(
        f"Train pEC50 vs measured log2_fc\nPearson={pearson:.2f}, Spearman={spearman:.2f}, N={len(sub):,}"
    )
    ax.set_xlabel("measured max log2_fc")
    ax.set_ylabel("train pEC50")
    ax.grid(alpha=0.2, linewidth=0.5)

    ax = axes[1, 1]
    csub = df[
        df["in_train"] & df["counter_pec50"].notna() & df["train_pec50"].notna()
    ].copy()
    csub["selectivity"] = csub["train_pec50"] - csub["counter_pec50"]
    ax.hist(
        csub["selectivity"],
        bins=45,
        color=COLORS["counter"],
        alpha=0.82,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(
        1.5, color="#c84b4b", linewidth=1.5, linestyle="--", label="selectivity >= 1.5"
    )
    ax.set_title(f"PXR selectivity from counter assay (N={len(csub):,})")
    ax.set_xlabel("train pEC50 - counter pEC50")
    ax.set_ylabel("compounds")
    ax.legend(frameon=True)

    savefig(fig, "auxiliary_data_distributions.png")


def plot_train_target_distribution(df: pd.DataFrame) -> None:
    train = df[df["in_train"] & df["train_pec50"].notna()].copy()
    train["selectivity"] = train["train_pec50"] - train["counter_pec50"]
    train["potent_selective"] = (
        train["train_pec50"].ge(6.0)
        & train["counter_pec50"].notna()
        & train["selectivity"].ge(1.5)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    bins = np.linspace(train["train_pec50"].min(), train["train_pec50"].max(), 55)
    ax.hist(
        train["train_pec50"],
        bins=bins,
        color=COLORS["train"],
        alpha=0.82,
        label=f"train pEC50 (N={len(train):,})",
    )
    potent = train[train["potent_selective"]]
    ax.hist(
        potent["train_pec50"],
        bins=bins,
        color="#c84b4b",
        alpha=0.9,
        label=f"potent/selective (N={len(potent):,})",
    )
    ax.axvline(6.0, color="#333333", linestyle="--", linewidth=1.2, label="pEC50 = 6")
    ax.set_title("Train pEC50 target distribution")
    ax.set_xlabel("pEC50")
    ax.set_ylabel("compounds")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    ax = axes[1]
    counter = train[train["counter_pec50"].notna()]
    ax.scatter(
        counter["train_pec50"],
        counter["selectivity"],
        s=11,
        alpha=0.38,
        color="#7a73a6",
        linewidths=0,
        label=f"train with counter assay (N={len(counter):,})",
    )
    ax.scatter(
        potent["train_pec50"],
        potent["selectivity"],
        s=28,
        alpha=0.9,
        color="#c84b4b",
        linewidths=0,
        label="potent/selective 46",
    )
    ax.axvline(6.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.axhline(1.5, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_title("Potency/selectivity definition")
    ax.set_xlabel("train pEC50")
    ax.set_ylabel("train pEC50 - counter pEC50")
    ax.legend(frameon=True)
    ax.grid(alpha=0.18, linewidth=0.5)

    savefig(fig, "train_target_distribution.png")


def plot_ro5_distributions(df: pd.DataFrame) -> None:
    plot_df = df[df["plot_role"].isin(["train", "test", "single_only"])].copy()
    panels = [
        ("amw", "Molecular weight", 500, "MW <= 500"),
        ("logp", "cLogP", 5, "LogP <= 5"),
        ("hba", "HBA", 10, "HBA <= 10"),
        ("hbd", "HBD", 5, "HBD <= 5"),
        ("num_rotatable_bonds", "Rotatable bonds", 10, "RB <= 10"),
        ("tpsa", "TPSA", 140, "TPSA <= 140"),
    ]
    role_order = ["single_only", "train", "test"]
    role_labels = {
        "single_only": "single-only aux",
        "train": "train",
        "test": "blind test",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
    for ax, (col, title, cutoff, cutoff_label) in zip(
        axes.ravel(), panels, strict=True
    ):
        values = plot_df[col].dropna()
        lo = values.quantile(0.005)
        hi = values.quantile(0.995)
        bins = np.linspace(lo, hi, 42)
        for role in role_order:
            vals = plot_df.loc[plot_df["plot_role"].eq(role), col].dropna()
            ax.hist(
                vals.clip(lo, hi),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                alpha=0.95,
                color=COLORS[role],
                label=f"{role_labels[role]} (N={len(vals):,})",
            )
        ax.axvline(
            cutoff, color="#333333", linestyle="--", linewidth=1.2, label=cutoff_label
        )
        ax.set_title(title)
        ax.set_ylabel("density")
        ax.grid(alpha=0.16, linewidth=0.5)
    axes[0, 0].legend(loc="best", frameon=True, fontsize=8)
    savefig(fig, "ro5_property_distributions.png")


def plot_ro5_violation_summary(df: pd.DataFrame) -> None:
    plot_df = df[df["plot_role"].isin(["train", "test", "single_only"])].copy()
    violations = pd.DataFrame(index=plot_df.index)
    violations["MW > 500"] = plot_df["amw"] > 500
    violations["LogP > 5"] = plot_df["logp"] > 5
    violations["HBA > 10"] = plot_df["hba"] > 10
    violations["HBD > 5"] = plot_df["hbd"] > 5
    plot_df["ro5_violations"] = violations.sum(axis=1)
    plot_df["veber_like_flags"] = (plot_df["num_rotatable_bonds"] > 10).astype(int) + (
        plot_df["tpsa"] > 140
    ).astype(int)

    rows = []
    for role, sub in plot_df.groupby("plot_role", observed=True):
        rows.append(
            {
                "role": role,
                "n": len(sub),
                "any_ro5_violation_pct": 100 * (sub["ro5_violations"] >= 1).mean(),
                "two_or_more_ro5_pct": 100 * (sub["ro5_violations"] >= 2).mean(),
                "rotb_or_tpsa_flag_pct": 100 * (sub["veber_like_flags"] >= 1).mean(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("role")

    label_map = {
        "single_only": "single-only aux",
        "train": "train",
        "test": "blind test",
    }
    summary["label"] = summary["role"].map(label_map)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(summary))
    width = 0.25
    metrics = [
        ("any_ro5_violation_pct", ">=1 Ro5 violation", "#777777"),
        ("two_or_more_ro5_pct", ">=2 Ro5 violations", "#c84b4b"),
        ("rotb_or_tpsa_flag_pct", "RB>10 or TPSA>140", "#4d8bc9"),
    ]
    for i, (col, label, color) in enumerate(metrics):
        bars = ax.bar(
            x + (i - 1) * width,
            summary[col],
            width=width,
            color=color,
            alpha=0.88,
            label=label,
        )
        ax.bar_label(
            bars, labels=[f"{v:.1f}%" for v in summary[col]], padding=3, fontsize=8
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.label}\nN={int(r.n):,}" for r in summary.itertuples()], fontsize=9
    )
    ax.set_ylabel("compounds (%)")
    ax.set_title("Drug-likeness flags by data role")
    ax.set_ylim(0, max(summary[[m[0] for m in metrics]].max()) * 1.25)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    savefig(fig, "ro5_violation_summary.png")


def add_structure_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["max_ring_atoms"] = [max_ring_atoms(smi) for smi in out["display_smiles"]]
    out["ro5_violations"] = (
        (out["amw"] > 500).astype(int)
        + (out["logp"] > 5).astype(int)
        + (out["hba"] > 10).astype(int)
        + (out["hbd"] > 5).astype(int)
    )
    return out


def take_evenly(df: pd.DataFrame, n: int, sort_col: str) -> pd.DataFrame:
    ordered = df.sort_values(sort_col).reset_index(drop=True)
    if len(ordered) <= n:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return ordered.iloc[idx].reset_index(drop=True)


def draw_compound_grid(
    rows: pd.DataFrame,
    out_name: str,
    legend_kind: str,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (330, 260),
) -> None:
    mols = []
    legends = []
    for row in rows.itertuples():
        mol = Chem.MolFromSmiles(row.display_smiles)
        if mol is None:
            continue
        rdCoordGen.AddCoords(mol)
        mols.append(mol)
        if legend_kind == "potent":
            selectivity = row.train_pec50 - row.counter_pec50
            legends.append(
                f"CID {row.compound_id}  pEC50 {row.train_pec50:.2f}\n"
                f"sel {selectivity:.1f}  MW {row.amw:.0f}  LogP {row.logp:.1f}"
            )
        elif legend_kind == "train":
            legends.append(
                f"CID {row.compound_id}  pEC50 {row.train_pec50:.2f}\n"
                f"MW {row.amw:.0f}  LogP {row.logp:.1f}  ring {row.max_ring_atoms}"
            )
        else:
            legends.append(
                f"CID {row.compound_id}\n"
                f"MW {row.amw:.0f}  LogP {row.logp:.1f}  ring {row.max_ring_atoms}"
            )
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=False,
    )
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    img.save(ASSET_DIR / out_name)


def plot_compound_example_grids(df: pd.DataFrame) -> None:
    flagged = add_structure_flags(df)
    train = flagged[flagged["plot_role"].eq("train")].copy()
    test = flagged[flagged["plot_role"].eq("test")].copy()

    typical_filter = (
        train["ro5_violations"].eq(0)
        & train["amw"].between(280, 430)
        & train["logp"].between(1.0, 4.5)
        & train["tpsa"].between(35, 115)
        & train["num_rotatable_bonds"].le(8)
    )
    typical_parts = [
        take_evenly(train[typical_filter & train["train_pec50"].lt(3.4)], 4, "amw"),
        take_evenly(
            train[
                typical_filter
                & train["train_pec50"].ge(4.2)
                & train["train_pec50"].le(5.0)
            ],
            4,
            "amw",
        ),
        take_evenly(train[typical_filter & train["train_pec50"].ge(5.8)], 4, "amw"),
    ]
    train_typical = pd.concat(typical_parts, ignore_index=True).head(12)

    fragments = take_evenly(
        train[train["amw"].lt(200) & train["train_pec50"].notna()],
        4,
        "amw",
    )
    macrocycles = take_evenly(
        train[train["max_ring_atoms"].ge(12) & train["train_pec50"].notna()],
        4,
        "amw",
    )
    ro5_edges = take_evenly(
        train[
            train["ro5_violations"].ge(2)
            | train["num_rotatable_bonds"].gt(10)
            | train["tpsa"].gt(140)
        ],
        4,
        "amw",
    )
    train_edges = (
        pd.concat([fragments, macrocycles, ro5_edges], ignore_index=True)
        .drop_duplicates("compound_id")
        .head(12)
    )
    if len(train_edges) < 12:
        edge_pool = train[
            train["amw"].gt(500)
            | train["amw"].lt(250)
            | train["max_ring_atoms"].ge(12)
            | train["ro5_violations"].ge(1)
        ]
        fill = take_evenly(
            edge_pool[~edge_pool["compound_id"].isin(train_edges["compound_id"])],
            12 - len(train_edges),
            "amw",
        )
        train_edges = pd.concat([train_edges, fill], ignore_index=True).head(12)

    test_filter = (
        test["ro5_violations"].eq(0)
        & test["amw"].between(280, 430)
        & test["logp"].between(1.0, 4.5)
        & test["tpsa"].between(35, 115)
        & test["num_rotatable_bonds"].le(8)
    )
    test_examples = take_evenly(test[test_filter], 12, "amw")

    draw_compound_grid(
        train_typical,
        "compound_examples_train_druglike.png",
        legend_kind="train",
    )
    draw_compound_grid(
        train_edges,
        "compound_examples_train_edge_cases.png",
        legend_kind="train",
        mols_per_row=3,
        sub_img_size=(380, 300),
    )
    draw_compound_grid(
        test_examples,
        "compound_examples_test_druglike.png",
        legend_kind="test",
    )

    potent = train[
        train["train_pec50"].ge(6.0)
        & train["counter_pec50"].notna()
        & (train["train_pec50"] - train["counter_pec50"]).ge(1.5)
    ].sort_values(["train_pec50", "compound_id"], ascending=[False, True])
    draw_compound_grid(
        potent,
        "compound_examples_potent46.png",
        legend_kind="potent",
        mols_per_row=10,
        sub_img_size=(300, 250),
    )


def main() -> None:
    df = load_master()
    emb = compute_or_load_umap(df)
    plot_all_compound_umap(emb)
    plot_auxiliary_data(df)
    plot_train_target_distribution(df)
    plot_ro5_distributions(df)
    plot_ro5_violation_summary(df)
    plot_compound_example_grids(df)


if __name__ == "__main__":
    main()
