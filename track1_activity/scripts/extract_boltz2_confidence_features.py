#!/usr/bin/env python
"""Extract Tier-1 structural features from Boltz-2 confidence NPZs.

Boltz-2 returns per-token pLDDT and per-token-pair PAE/PDE. Tokens = 434
PXR residues (UniProt O75469) followed by ligand_atom_count atom tokens.
Current DB scalars aggregate over the whole 457x457 block; we re-aggregate
separately for:

  protein all           -- residues 0..433
  ligand all            -- atoms 434..(434+N-1)
  pocket core           -- 13 PXR core pocket residues (constants.py)
  protein-ligand cross  -- cross block of PAE/PDE matrices
  pocket-ligand cross   -- rows=pocket, cols=ligand
  intra-ligand          -- ligand-only block

Per group we take mean / std / min / max + (for plddt) 10th percentile.
Also stores per-residue pLDDT for each of the 13 pocket residues as
individual features (for possible key-contact signal).

Output:
  - data/boltz2_confidence_features.parquet  (one row per compound_id)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.joinpath("track1_activity/boltz2/src")))

from boltz2.constants import PXR_CORE_POCKET_RESIDUES  # noqa: E402

DB_KW = dict(host="/tmp", port=5433, dbname="pxr_challenge")
OUT_PATH = ROOT.joinpath("data", "boltz2_confidence_features.parquet")
PROTEIN_LEN = 434
# Residue numbering is 1-based in UniProt; arrays are 0-based
POCKET_IDX = np.array([r - 1 for r in PXR_CORE_POCKET_RESIDUES], dtype=int)


def safe_stats(arr: np.ndarray, prefix: str, pctl: bool = False) -> dict:
    """mean/std/min/max; optional 10th percentile for pLDDT."""
    if arr.size == 0:
        nan = float("nan")
        out = {
            f"{prefix}_mean": nan,
            f"{prefix}_std": nan,
            f"{prefix}_min": nan,
            f"{prefix}_max": nan,
        }
        if pctl:
            out[f"{prefix}_p10"] = nan
        return out
    out = {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }
    if pctl:
        out[f"{prefix}_p10"] = float(np.percentile(arr, 10))
    return out


def compute_features(
    plddt_path: str, pae_path: str, pde_path: str, lac: int
) -> dict | None:
    """Return a flat dict of features, or None if shapes are inconsistent."""
    plddt = np.load(plddt_path)["plddt"]
    pae = np.load(pae_path)["pae"]
    pde = np.load(pde_path)["pde"]
    n = PROTEIN_LEN + lac
    if plddt.shape[0] != n or pae.shape != (n, n) or pde.shape != (n, n):
        return None

    prot = slice(0, PROTEIN_LEN)
    lig = slice(PROTEIN_LEN, n)

    feats: dict = {}
    # pLDDT stats
    feats.update(safe_stats(plddt[prot], "plddt_protein", pctl=True))
    feats.update(safe_stats(plddt[lig], "plddt_ligand", pctl=True))
    feats.update(safe_stats(plddt[POCKET_IDX], "plddt_pocket", pctl=True))

    # PAE stats (cross-blocks)
    pae_pl = pae[prot, lig]  # (434, N)
    pae_pocketl = pae[POCKET_IDX][:, lig]  # (13, N)
    feats.update(safe_stats(pae_pl, "pae_protein_ligand"))
    feats.update(safe_stats(pae_pocketl, "pae_pocket_ligand"))

    # PDE stats (cross + intra-ligand)
    pde_pocketl = pde[POCKET_IDX][:, lig]
    pde_intralig = pde[lig, lig]
    feats.update(safe_stats(pde_pocketl, "pde_pocket_ligand"))
    feats.update(safe_stats(pde_intralig, "pde_intra_ligand"))

    # Per-residue pLDDT for each of 13 pocket residues
    for resnum, idx in zip(PXR_CORE_POCKET_RESIDUES, POCKET_IDX):
        feats[f"plddt_res{resnum}"] = float(plddt[idx])

    return feats


def main() -> None:
    with psycopg2.connect(**DB_KW) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT compound_id, ligand_atom_count,
                   plddt_npz_path, pae_npz_path, pde_npz_path
            FROM compound_boltz2
            WHERE plddt_npz_path IS NOT NULL
              AND pae_npz_path IS NOT NULL
              AND pde_npz_path IS NOT NULL
              AND ligand_atom_count IS NOT NULL
            ORDER BY compound_id
            """
        )
        rows = cur.fetchall()
    print(f"Found {len(rows)} compounds with confidence NPZs")

    records = []
    skipped = 0
    for cid, lac, plddt_p, pae_p, pde_p in tqdm(rows):
        try:
            feats = compute_features(plddt_p, pae_p, pde_p, lac)
        except Exception as e:  # noqa: BLE001 - want to log & continue
            print(f"  ERROR cid={cid}: {e}")
            skipped += 1
            continue
        if feats is None:
            skipped += 1
            continue
        feats["compound_id"] = cid
        records.append(feats)
    print(
        f"Extracted {len(records)} rows; {skipped} skipped "
        f"(shape mismatch or read error)"
    )

    df = pd.DataFrame(records).set_index("compound_id").sort_index()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Saved: {OUT_PATH}  (shape={df.shape})")
    print("Columns sample:", df.columns.tolist()[:10], "...")


if __name__ == "__main__":
    main()
