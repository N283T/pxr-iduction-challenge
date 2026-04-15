#!/usr/bin/env -S pixi run python
"""Render 2D molecule grids for the compound groups flagged in 01-04.

Static PNG grids let us eyeball whether the numerical outliers are really
the chemotypes we suspect (cardenolides, peptidic dimers, surfactants,
reporter artifacts) without having to open marimo.

Outputs (docs/figures/):
  - eda_redo_mols_oversize.png              - largest train compounds vs test max
  - eda_redo_mols_pose_outliers.png         - pose_distance > 4 A, sorted by pEC50
  - eda_redo_mols_train_only_clusters.png   - HDBSCAN train-only cluster examples
  - eda_redo_mols_non_selective_hi.png      - counter_pec50 ~ train_pec50, high pEC50
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

from eda_redo import draw_mol_grid_png, load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")


def _grid(df: pd.DataFrame, legend_fn, out_name: str, n: int = 12) -> None:
    smiles = df["smiles"].head(n).tolist()
    legends = [legend_fn(row) for _, row in df.head(n).iterrows()]
    path = FIG_DIR.joinpath(out_name)
    draw_mol_grid_png(smiles, legends, path, mols_per_row=4)
    print(f"[05] wrote {path}  (N={len(smiles)})")


def main() -> None:
    master = load_master()

    # ------------------------------------------------------------------
    # 1. Largest compounds (oversize vs test)
    # ------------------------------------------------------------------
    oversize = pd.read_parquet(DATA_DIR.joinpath("01_size_flagged.parquet"))
    oversize = oversize.sort_values("num_heavy_atoms", ascending=False)
    _grid(
        oversize,
        lambda r: (
            f"cid {int(r['compound_id'])}\nHA={int(r['num_heavy_atoms'])}  pEC50={r['train_pec50']:.2f}"
        ),
        "eda_redo_mols_oversize.png",
        n=12,
    )

    # ------------------------------------------------------------------
    # 2. Pose outliers (pocket dist > 4 A), ordered by pEC50 desc so the most
    #    dangerous-looking "active + weird pose" compounds come first.
    # ------------------------------------------------------------------
    pose = pd.read_parquet(DATA_DIR.joinpath("02_pose_outliers.parquet"))
    pose = pose.sort_values("train_pec50", ascending=False, na_position="last")
    _grid(
        pose,
        lambda r: (
            f"cid {int(r['compound_id'])}\n"
            f"dist={r['b2_pocket_distance_a']:.1f}A  pEC50={r['train_pec50']:.2f}"
        ),
        "eda_redo_mols_pose_outliers.png",
        n=12,
    )

    # ------------------------------------------------------------------
    # 3. Train-only HDBSCAN clusters: show 4 representative compounds per cluster
    # ------------------------------------------------------------------
    emb = pd.read_parquet(DATA_DIR.joinpath("03_chemotype_embedding.parquet"))
    clusters = pd.read_parquet(DATA_DIR.joinpath("03_chemotype_clusters.parquet"))
    train_only_ids = clusters[(clusters["n_test"] == 0) & (clusters["cluster"] >= 0)][
        "cluster"
    ].tolist()
    picks = []
    for cid in train_only_ids:
        members = emb[(emb["cluster"] == cid) & (emb["split"] == "train")].copy()
        # Pick 4: the two highest-pEC50 and two nearest the cluster center.
        members = members.sort_values("train_pec50", ascending=False)
        top = members.head(2)
        mid = members.iloc[len(members) // 2 : len(members) // 2 + 2]
        picks.append(pd.concat([top, mid]))
    pick_df = pd.concat(picks).reset_index(drop=True)
    _grid(
        pick_df,
        lambda r: (
            f"cluster {int(r['cluster'])}  cid {int(r['compound_id'])}\n"
            f"HA={int(r['num_heavy_atoms'])}  pEC50={r['train_pec50']:.2f}"
        ),
        "eda_redo_mols_train_only_clusters.png",
        n=len(pick_df),
    )

    # ------------------------------------------------------------------
    # 4. Non-selective with high pEC50 (counter_pec50 close to train_pec50).
    #    These are the suspected reporter-artifact "hits" we want to eyeball.
    # ------------------------------------------------------------------
    ns = pd.read_parquet(DATA_DIR.joinpath("04_non_selective.parquet"))
    hi_ns = ns[ns["train_pec50"] >= 5.0].sort_values("train_pec50", ascending=False)
    _grid(
        hi_ns,
        lambda r: (
            f"cid {int(r['compound_id'])}\n"
            f"pEC50={r['train_pec50']:.2f}  cnt={r['counter_pec50']:.2f}"
        ),
        "eda_redo_mols_non_selective_hi.png",
        n=12,
    )

    # ------------------------------------------------------------------
    # 5. The 67 potent train compounds: are they mostly in the big cluster?
    #    Helpful for seeing what a "real" PXR inducer looks like in this set.
    # ------------------------------------------------------------------
    potent = master[master["train_is_potent"].fillna(False)].copy()
    potent = potent.sort_values("train_pec50", ascending=False)
    _grid(
        potent,
        lambda r: f"cid {int(r['compound_id'])}\npEC50={r['train_pec50']:.2f}",
        "eda_redo_mols_top_potent.png",
        n=12,
    )
    print(f"[05] total potent train compounds: {len(potent)}")


if __name__ == "__main__":
    main()
