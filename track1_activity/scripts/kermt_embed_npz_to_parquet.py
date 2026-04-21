"""Convert KERMT fingerprint output to the parquet format expected by
``run_train.py --feature kermt_pretrain_embed``.

KERMT's ``main.py fingerprint`` (and our ``kermt_fingerprint_patched.py``
wrapper) writes a compressed npz with a single ``fps`` array of shape
(n_compounds, emb_dim). It strips compound_ids, so row order is the only
handle back to compound_id. Our canonical input
``data/kermt/pretrain_all.csv`` is already sorted by compound_id ascending,
and the SMILES-only CSV derived from it preserves that order, so row ``i``
of the npz corresponds to the compound_id in row ``i + 1`` of
pretrain_all.csv (the header adds the +1).

Reads:
  - data/kermt/pretrain_all.csv   (compound_id ordering)
  - data/kermt/embeddings.npz     (KERMT native output)
Writes:
  - data/kermt_pretrain_embed.parquet
    (index = compound_id, columns = emb_0000..emb_{N-1})
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ID_CSV = REPO_ROOT.joinpath("data", "kermt", "pretrain_all.csv")
NPZ_PATH = REPO_ROOT.joinpath("data", "kermt", "embeddings.npz")
OUT_PATH = REPO_ROOT.joinpath("data", "kermt_pretrain_embed.parquet")


def load_kermt_embeddings(npz_path: Path) -> np.ndarray:
    """Load the KERMT fingerprint npz.

    KERMT stores the fingerprint tensor under a single key ``fps``
    (see main.py: ``np.savez_compressed(args.output_path, fps=feas)``).
    We tolerate a few alternative key names for forward-compatibility.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run run_kermt_embed_extract.sh first."
        )
    data = np.load(npz_path)
    keys = list(data.keys())
    print(f"NPZ keys: {keys}")
    for key in ("fps", "embedding", "fingerprint", "features", "arr_0"):
        if key in data:
            arr = data[key]
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected a 2-D embedding array for key '{key}', "
                    f"got shape {arr.shape}"
                )
            return arr.astype(np.float32)
    raise ValueError(f"No recognised embedding key in {npz_path}. Keys found: {keys}")


def main() -> None:
    id_df = pd.read_csv(ID_CSV)
    if "compound_id" not in id_df.columns:
        raise ValueError(
            f"{ID_CSV} must contain a 'compound_id' column; got {id_df.columns.tolist()}"
        )
    compound_ids = id_df["compound_id"].astype(int).to_numpy()
    print(f"Loaded {len(compound_ids)} compound_ids from {ID_CSV}")

    emb = load_kermt_embeddings(NPZ_PATH)
    print(f"Loaded embeddings shape {emb.shape} dtype {emb.dtype}")

    if len(compound_ids) != emb.shape[0]:
        raise ValueError(
            f"Row count mismatch: {len(compound_ids)} compound_ids "
            f"vs {emb.shape[0]} embedding rows. KERMT may have dropped "
            f"some compounds (e.g. unparseable SMILES). Inspect "
            f"{NPZ_PATH}."
        )

    nan_count = int(np.isnan(emb).sum())
    if nan_count > 0:
        raise ValueError(
            f"Found {nan_count} NaN values in KERMT embeddings; refusing to "
            f"write parquet. Check KERMT run for silent failures."
        )

    cols = [f"emb_{i:04d}" for i in range(emb.shape[1])]
    df = pd.DataFrame(
        emb,
        index=pd.Index(compound_ids, name="compound_id"),
        columns=cols,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
