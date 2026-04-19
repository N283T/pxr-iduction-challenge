"""Split structures/gator/train.pkl into 5 UMAP CV fold pkls.

Uses the canonical UMAP+Morgan+Jaccard+k=50+seed=42 split shared
across the project. Aligns compound_ids across:
  - train_activity (source of UMAP clusters and pEC50)
  - structures/gator/train.pkl items (indexed by PDB_ID = compound_id)

For each fold k we write:
  structures/gator/folds/fold{k}_train.pkl
  structures/gator/folds/fold{k}_val.pkl

`fold{k}_val.pkl` has a parallel ids_val_{k}.json so downstream can
recover compound_ids for OOF collation without re-parsing PDB_IDs.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402


def main() -> None:
    gator_root = REPO_ROOT.joinpath("structures", "gator")
    pkl_path = gator_root.joinpath("train.pkl")
    folds_dir = gator_root.joinpath("folds")
    folds_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        items = pickle.load(f)
    print(f"Loaded {len(items)} items from {pkl_path}")

    # Parse compound_ids from item['id'] ("00001_UNL" -> 1)
    item_cids = np.array(
        [int(it["id"].split("_")[0]) for it in items], dtype=np.int64
    )
    print(f"Item compound_id range: {item_cids.min()}..{item_cids.max()}")

    conn = psycopg2.connect(**DB_PARAMS)
    ta = pd.read_sql(
        "SELECT compound_id, pec50 FROM train_activity ORDER BY id", conn
    )
    cmp_df = pd.read_sql(
        "SELECT id, std_smiles FROM compounds", conn
    ).set_index("id")
    conn.close()

    # Full train_activity has 4140 rows; our train.pkl has 4139 (one Boltz
    # preprocess failure). Build an item lookup and subset ta accordingly.
    cid_to_item_idx = {int(c): i for i, c in enumerate(item_cids)}
    ta_cids = ta["compound_id"].to_numpy()
    keep_mask = np.array([int(c) in cid_to_item_idx for c in ta_cids])
    ta_kept = ta[keep_mask].reset_index(drop=True)
    print(
        f"train_activity rows kept (intersection with train.pkl): "
        f"{len(ta_kept)}/{len(ta)}"
    )

    smiles_list = [
        cmp_df.loc[int(c), "std_smiles"] for c in ta_kept["compound_id"]
    ]

    # Canonical 5-fold UMAP split
    splits = umap_split_indices(smiles_list, n_splits=5, seed=42)
    print(f"Computed {len(splits)} UMAP folds")

    for k, (train_idx_in_ta, val_idx_in_ta) in enumerate(splits):
        # Map back through ta_kept row ordering -> item index in pkl
        def _map(idx_in_ta: np.ndarray) -> list[int]:
            cids = ta_kept["compound_id"].iloc[idx_in_ta].to_numpy()
            return [cid_to_item_idx[int(c)] for c in cids]

        tr = _map(train_idx_in_ta)
        vl = _map(val_idx_in_ta)
        tr_items = [items[i] for i in tr]
        vl_items = [items[i] for i in vl]

        with open(folds_dir.joinpath(f"fold{k}_train.pkl"), "wb") as f:
            pickle.dump(tr_items, f)
        with open(folds_dir.joinpath(f"fold{k}_val.pkl"), "wb") as f:
            pickle.dump(vl_items, f)
        val_cids = [int(it["id"].split("_")[0]) for it in vl_items]
        with open(folds_dir.joinpath(f"fold{k}_val_cids.json"), "w") as f:
            json.dump(val_cids, f)

        print(
            f"fold{k}: train={len(tr_items)}, val={len(vl_items)} "
            f"-> {folds_dir.joinpath(f'fold{k}_*.pkl')}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
