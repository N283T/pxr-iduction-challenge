"""Boltz-like pooling: replace core-pocket interface z with ALL
protein x ligand cross pairs (matches Boltz-2 affinity head's
cross_pair_mask in affinity.py:121-125, minus lig x lig diagonal).

Output shape per compound (same 1024 dim as 01_pool_embeddings.py
for a clean A/B comparison):
  s_prot_mean (384)
  s_lig_mean  (384)
  z_xp_mean   (128) = mean over (434 PXR residues) x (ligand atoms) pairs
  z_xp_max    (128) = max  over the same pairs

Result -> data/boltz_affhead/pooled_allpairs.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track2_structure", "src", "boltz2"))
)
from constants import PXR_SEQUENCE  # noqa: E402
from data import DB_PARAMS  # noqa: E402


PROTEIN_N_RES = len(PXR_SEQUENCE)  # 434
OUT_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled_allpairs.parquet")


def pool_one(
    npz_path: Path, n_ligand_atoms: int
) -> dict[str, np.ndarray] | None:
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        print(f"  skip {npz_path.name}: load error {e}")
        return None
    s = data["s"][0]
    z = data["z"][0]
    T = s.shape[0]
    expected = PROTEIN_N_RES + n_ligand_atoms
    if T != expected:
        print(
            f"  warn {npz_path.name}: T={T} != expected {expected}"
        )

    prot_slice = slice(0, PROTEIN_N_RES)
    lig_slice = slice(PROTEIN_N_RES, T)

    s_prot = s[prot_slice].mean(axis=0).astype(np.float32)
    s_lig = s[lig_slice].mean(axis=0).astype(np.float32)

    # All protein x ligand cross pairs (Boltz-2's rec x lig mask)
    # z_xp shape: (PROTEIN_N_RES, L, 128)
    z_xp = z[prot_slice, lig_slice, :]
    if z_xp.size == 0:
        return None
    z_xp_flat = z_xp.reshape(-1, z_xp.shape[-1])
    z_xp_mean = z_xp_flat.mean(axis=0).astype(np.float32)
    z_xp_max = z_xp_flat.max(axis=0).astype(np.float32)

    return dict(
        s_prot_mean=s_prot,
        s_lig_mean=s_lig,
        z_xp_mean=z_xp_mean,
        z_xp_max=z_xp_max,
    )


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(**DB_PARAMS)
    rows = pd.read_sql(
        """
        SELECT compound_id, embeddings_npz_path, ligand_atom_count
        FROM compound_boltz2
        WHERE embeddings_npz_path IS NOT NULL
          AND preprocessing_failed = FALSE
          AND ligand_atom_count IS NOT NULL
        ORDER BY compound_id
        """,
        conn,
    )
    conn.close()
    print(f"Candidates: {len(rows)}")

    records = []
    skipped = 0
    for i, r in enumerate(rows.itertuples(index=False), 1):
        pooled = pool_one(Path(r.embeddings_npz_path), int(r.ligand_atom_count))
        if pooled is None:
            skipped += 1
            continue
        row = {"compound_id": int(r.compound_id)}
        for name, vec in pooled.items():
            for j, v in enumerate(vec):
                row[f"{name}_{j:03d}"] = float(v)
        records.append(row)
        if i % 500 == 0 or i <= 5:
            print(f"  [{i}/{len(rows)}] cid={r.compound_id} pooled")

    df = pd.DataFrame.from_records(records)
    print(f"\nWriting {len(df)} rows -> {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False, compression="zstd")
    print(f"  columns = {len(df.columns) - 1}")
    print(f"  skipped = {skipped}")


if __name__ == "__main__":
    main()
