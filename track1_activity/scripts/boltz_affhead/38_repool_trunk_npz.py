"""Re-pool raw Boltz trunk s/z NPZ files into region-level summaries.

Input is ``compound_boltz2_trunk_fast.source_npz_path`` for both:

* rcycle=3 full-run rows (4652)
* rcycle=1 embeddings-only rows (8482)

This script intentionally uses only trunk tensors, not pose/confidence/affinity
outputs, so the resulting feature can cover the full 13,134-row trunk-fast set.

Default variant: ``region_zstats``.

Output:
  data/boltz_affhead/repooled_trunk_region_zstats.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "boltz2", "src", "boltz2"))
)

from constants import PXR_CORE_POCKET_RESIDUES, PXR_SEQUENCE  # noqa: E402
from data import get_engine  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "boltz_affhead")

PROTEIN_N_RES = len(PXR_SEQUENCE)
CORE_IDX = np.array([r - 1 for r in PXR_CORE_POCKET_RESIDUES], dtype=np.int64)

# Broad PXR regions in UniProt residue numbering. These are intentionally
# coarse; the goal is to preserve regional trunk signal without a 434-residue
# feature explosion.
REGION_RANGES: dict[str, tuple[int, int]] = {
    "nterm": (1, 140),
    "lbd_entrance": (141, 210),
    "lbd_body": (211, 330),
    "h11_h12": (331, 434),
}


def region_indices() -> dict[str, np.ndarray]:
    """Return 0-indexed protein token indices for coarse PXR regions."""
    regions: dict[str, np.ndarray] = {
        name: np.arange(start - 1, end, dtype=np.int64)
        for name, (start, end) in REGION_RANGES.items()
    }
    regions["core_pocket"] = CORE_IDX
    return regions


def summarize_block(prefix: str, values: np.ndarray) -> dict[str, float]:
    """Summarize a 2D block as mean/std/q10/q90 per embedding dimension."""
    if values.ndim != 2:
        raise ValueError(f"{prefix}: expected 2D array, got shape {values.shape}")
    stats = {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
    }
    row: dict[str, float] = {}
    for stat_name, vec in stats.items():
        for i, val in enumerate(vec.astype(np.float32)):
            row[f"{prefix}_{stat_name}_{i:03d}"] = float(val)
    return row


def summarize_block_selected(
    prefix: str,
    values: np.ndarray,
    selected_stats: tuple[str, ...],
) -> dict[str, float]:
    """Summarize a 2D block with an explicit subset of statistics."""
    if values.ndim != 2:
        raise ValueError(f"{prefix}: expected 2D array, got shape {values.shape}")
    stats = {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
    }
    row: dict[str, float] = {}
    for stat_name in selected_stats:
        vec = stats[stat_name]
        for i, val in enumerate(vec.astype(np.float32)):
            row[f"{prefix}_{stat_name}_{i:03d}"] = float(val)
    return row


def pool_npz_region_zstats(
    npz_path: Path,
    protein_n_res: int = PROTEIN_N_RES,
    regions: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Pool one raw Boltz trunk NPZ into region-level z and token summaries."""
    regions = regions or region_indices()
    data = np.load(npz_path, allow_pickle=False)
    s = data["s"][0]
    z = data["z"][0]
    token_count = s.shape[0]
    if token_count <= protein_n_res:
        raise ValueError(f"{npz_path}: token_count={token_count} <= {protein_n_res}")
    lig_slice = slice(protein_n_res, token_count)
    lig_tokens = token_count - protein_n_res

    row: dict[str, float] = {
        "ligand_tokens": float(lig_tokens),
    }
    # Keep s-token summaries compact. The expensive part of this feature is
    # the protein-ligand z representation; broad s-region summaries would add
    # thousands of dimensions while mostly retesting old s-only ablations.
    row.update(summarize_block_selected("s_lig", s[lig_slice, :], ("mean", "std")))
    core_idx = CORE_IDX[(CORE_IDX >= 0) & (CORE_IDX < protein_n_res)]
    if core_idx.size:
        row.update(summarize_block_selected("s_core_pocket", s[core_idx, :], ("mean",)))

    for region_name, idx in regions.items():
        valid_idx = idx[(idx >= 0) & (idx < protein_n_res)]
        if valid_idx.size == 0:
            continue
        # Protein-region x ligand cross-pairs: (R, L, 128) -> (R*L, 128)
        z_block = z[
            np.ix_(
                valid_idx, np.arange(protein_n_res, token_count), np.arange(z.shape[-1])
            )
        ]
        z_flat = z_block.reshape(-1, z.shape[-1])
        row.update(summarize_block(f"z_{region_name}", z_flat))

    return row


def fetch_trunk_sources(limit: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT compound_id, recycling_steps, source_npz_path
          FROM compound_boltz2_trunk_fast
         WHERE source_npz_path IS NOT NULL
         ORDER BY compound_id
    """
    if limit is not None:
        sql += f"\n         LIMIT {int(limit)}"
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pool all rows into a dense float32 DataFrame without dict-list blowup."""
    if df.empty:
        return pd.DataFrame()

    first = df.iloc[0]
    first_pooled = pool_npz_region_zstats(Path(first["source_npz_path"]))
    feature_cols = list(first_pooled.keys())
    features = np.empty((len(df), len(feature_cols)), dtype=np.float32)
    compound_ids = df["compound_id"].to_numpy(dtype=np.int64)
    recycling_steps = df["recycling_steps"].to_numpy(dtype=np.int16)

    features[0, :] = np.array([first_pooled[c] for c in feature_cols], dtype=np.float32)
    print(
        f"  [1/{len(df)}] cid={int(first['compound_id'])} "
        f"rcycle={int(first['recycling_steps'])} cols={len(feature_cols)}"
    )

    for i, row in enumerate(df.iloc[1:].itertuples(index=False), 2):
        npz_path = Path(row.source_npz_path)
        pooled = pool_npz_region_zstats(npz_path)
        if list(pooled.keys()) != feature_cols:
            raise ValueError(
                f"Feature columns changed at compound_id={row.compound_id}"
            )
        features[i - 1, :] = np.array(
            [pooled[c] for c in feature_cols], dtype=np.float32
        )
        if i <= 3 or i % 250 == 0 or i == len(df):
            print(
                f"  [{i}/{len(df)}] cid={row.compound_id} "
                f"rcycle={row.recycling_steps} cols={len(feature_cols)}"
            )

    out_df = pd.DataFrame(features, columns=feature_cols)
    out_df.insert(0, "recycling_steps", recycling_steps)
    out_df.insert(0, "compound_id", compound_ids)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR.joinpath("repooled_trunk_region_zstats.parquet"),
    )
    args = parser.parse_args()

    df = fetch_trunk_sources(limit=args.limit)
    print(f"Sources: {len(df)}")
    out_df = build_feature_frame(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False, compression="zstd")
    print(
        f"Wrote {out_df.shape[0]} rows x {out_df.shape[1] - 2} features -> {args.out}"
    )


if __name__ == "__main__":
    main()
