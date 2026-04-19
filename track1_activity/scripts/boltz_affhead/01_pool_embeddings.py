"""Pool Boltz-2 trunk embeddings per compound into fixed-size vectors.

Boltz-2 stores trunk output per complex as an npz with:
  s: (1, T, 384)        -- per-token single embedding
  z: (1, T, T, 128)     -- pairwise embedding

T = 434 (PXR residues) + L (ligand atoms, from compound_boltz2.ligand_atom_count).
Protein tokens occupy indices [0, 434); ligand tokens occupy [434, 434 + L).

We compute three pooling strategies per compound:
  1. s_prot_mean      : mean of s over protein tokens    -> (384,)
  2. s_lig_mean       : mean of s over ligand tokens     -> (384,)
  3. z_interface_pool : mean + max of z at
                        (core_pocket_residues x ligand_tokens) -> (256,)

Core pocket residues are the 13 contact residues from
track2_structure/src/boltz2/constants.py (UniProt numbering, 1-based).
Token index = residue_number - 1.

Output: structures/gator/... no wait, this is a Boltz-path experiment.
Output: data/boltz_affhead/pooled.parquet keyed by compound_id with
columns s_prot_mean_{0..383}, s_lig_mean_{0..383},
z_if_mean_{0..127}, z_if_max_{0..127}.
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
from constants import PXR_CORE_POCKET_RESIDUES, PXR_SEQUENCE  # noqa: E402
from data import DB_PARAMS  # noqa: E402


PROTEIN_N_RES = len(PXR_SEQUENCE)  # 434
# Token index for each core pocket residue (0-based)
CORE_POCKET_IDX = np.asarray(
    [r - 1 for r in PXR_CORE_POCKET_RESIDUES], dtype=np.int64
)

OUT_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled.parquet")


def pool_one(
    npz_path: Path, n_ligand_atoms: int
) -> dict[str, np.ndarray] | None:
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        print(f"  skip {npz_path.name}: load error {e}")
        return None
    s = data["s"][0]  # (T, 384)
    z = data["z"][0]  # (T, T, 128)
    T = s.shape[0]
    expected = PROTEIN_N_RES + n_ligand_atoms
    if T != expected:
        print(
            f"  warn {npz_path.name}: T={T} != expected {expected} "
            f"(protein {PROTEIN_N_RES} + ligand {n_ligand_atoms})"
        )
        # Still proceed with the actual token split
    prot_slice = slice(0, PROTEIN_N_RES)
    lig_slice = slice(PROTEIN_N_RES, T)

    s_prot = s[prot_slice].mean(axis=0).astype(np.float32)
    s_lig = s[lig_slice].mean(axis=0).astype(np.float32)

    # Interface z: pocket residues x ligand atoms
    # z[i, j, :] = pair (i, j). Take z[pocket_idx[:, None], lig_idx[None, :]]
    pocket_ok = CORE_POCKET_IDX < PROTEIN_N_RES
    pocket_idx = CORE_POCKET_IDX[pocket_ok]
    lig_idx = np.arange(PROTEIN_N_RES, T, dtype=np.int64)
    if lig_idx.size == 0:
        return None
    z_if = z[pocket_idx[:, None], lig_idx[None, :], :]  # (P, L, 128)
    z_if_mean = z_if.reshape(-1, z_if.shape[-1]).mean(axis=0).astype(np.float32)
    z_if_max = z_if.reshape(-1, z_if.shape[-1]).max(axis=0).astype(np.float32)

    return dict(
        s_prot_mean=s_prot,
        s_lig_mean=s_lig,
        z_if_mean=z_if_mean,
        z_if_max=z_if_max,
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
