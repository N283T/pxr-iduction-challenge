#!/usr/bin/env -S pixi run python
"""Render all drop-candidate compounds as 2D grids.

Two populations of structural outliers:
  1. Big tail  - n_out >= 5 using train p1/p99 on 11 descriptors
                 (macrolide/peptide/rapamycin-like, already hard drops).
  2. Small tail - HA <= 10 (amino acids, ethanolamine-class fragments,
                 standard reagents; cannot occupy PXR LBD).

Output:
  - eda_redo_07_drop_big_tail.png      - big outliers (one PNG)
  - eda_redo_07_drop_small_tail.png    - small fragments (one PNG)
  - eda_redo_07_drop_candidates.parquet - combined compound list
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from eda_redo import draw_mol_grid_png, load_master

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

N_OUT_THRESHOLD = 5
HA_SMALL_THRESHOLD = 10


def _legend(r: pd.Series) -> str:
    pec = r.get("train_pec50")
    pec_str = f"{pec:.2f}" if pd.notna(pec) else "NA"
    return (
        f"cid {int(r['compound_id'])}  HA={int(r['num_heavy_atoms'])}  "
        f"MW={r['amw']:.0f}\npEC50={pec_str}"
    )


def main() -> None:
    master = load_master()
    tr = master[master["split"] == "train"].copy()
    scorecard = pd.read_parquet(DATA_DIR.joinpath("06_outlier_scorecard.parquet"))
    big_ids = scorecard[scorecard["n_out"] >= N_OUT_THRESHOLD]["compound_id"].tolist()
    big = tr[tr["compound_id"].isin(big_ids)].copy()
    big = big.sort_values(["num_heavy_atoms", "amw"], ascending=[False, False])
    print(f"[07] big tail (n_out >= {N_OUT_THRESHOLD}): N={len(big)}")

    small = tr[tr["num_heavy_atoms"] <= HA_SMALL_THRESHOLD].copy()
    small = small.sort_values(["num_heavy_atoms", "amw"], ascending=[True, True])
    print(f"[07] small tail (HA <= {HA_SMALL_THRESHOLD}): N={len(small)}")

    overlap = set(big["compound_id"]) & set(small["compound_id"])
    combined_ids = list(set(big["compound_id"]) | set(small["compound_id"]))
    print(f"[07] combined (union): N={len(combined_ids)}  overlap={len(overlap)}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Big tail grid (bigger subimages because the structures are complex)
    # ------------------------------------------------------------------
    big_smiles = big["smiles"].tolist()
    big_legends = [_legend(r) for _, r in big.iterrows()]
    big_path = FIG_DIR.joinpath("eda_redo_07_drop_big_tail.png")
    draw_mol_grid_png(
        big_smiles,
        big_legends,
        big_path,
        mols_per_row=4,
        sub_img_size=(360, 280),
    )
    print(f"[07] wrote {big_path}  (N={len(big_smiles)})")

    # ------------------------------------------------------------------
    # Small tail grid - many compounds, so use smaller subimages + wider grid
    # ------------------------------------------------------------------
    small_smiles = small["smiles"].tolist()
    small_legends = [_legend(r) for _, r in small.iterrows()]
    small_path = FIG_DIR.joinpath("eda_redo_07_drop_small_tail.png")
    draw_mol_grid_png(
        small_smiles,
        small_legends,
        small_path,
        mols_per_row=8,
        sub_img_size=(200, 160),
    )
    print(f"[07] wrote {small_path}  (N={len(small_smiles)})")

    # ------------------------------------------------------------------
    # Combined candidate parquet
    # ------------------------------------------------------------------
    def _tag(cid: int) -> str:
        b = cid in set(big["compound_id"])
        s = cid in set(small["compound_id"])
        if b and s:
            return "both"
        return "big_tail" if b else "small_tail"

    combined = tr[tr["compound_id"].isin(combined_ids)].copy()
    combined["drop_reason"] = combined["compound_id"].map(_tag)
    combined = combined.sort_values(
        ["drop_reason", "num_heavy_atoms"], ascending=[True, False]
    )
    keep_cols = [
        "compound_id",
        "drop_reason",
        "num_heavy_atoms",
        "amw",
        "logp",
        "tpsa",
        "num_rotatable_bonds",
        "train_pec50",
        "train_emax_vs_pos_ctrl",
        "counter_pec50",
        "counter_minus_train_pec50",
        "b2_pocket_distance_a",
        "inchikey",
        "smiles",
    ]
    out = combined[keep_cols]
    out_path = DATA_DIR.joinpath("07_drop_candidates.parquet")
    out.to_parquet(out_path, index=False)
    print(f"[07] wrote {out_path}")
    print()
    print("[07] combined drop candidates by reason:")
    print(out["drop_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
