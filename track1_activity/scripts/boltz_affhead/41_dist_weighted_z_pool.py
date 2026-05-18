#!/usr/bin/env -S pixi run python
"""Distance-weighted pooling of Boltz trunk pair representation.

This is a stronger distogram experiment than tabular distance summaries:
instead of appending pose-distance features to pooled trunk features, use the
predicted pose distances to re-pool the trunk ``z`` protein-ligand pair tensor.

The motivating analogue is Boltz-2's affinity head, where a distogram conditions
the pair representation before PairFormer. This script keeps the head tabular
and cheap, but makes the distance interact directly with ``z``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "boltz2", "src", "boltz2"))
)

from constants import PXR_CORE_POCKET_RESIDUES, PROTEIN_CHAIN_ID, PXR_SEQUENCE  # noqa: E402
from data import DB_PARAMS  # noqa: E402

OUT_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "dist_weighted_z_pool.parquet")
PROTEIN_N_RES = len(PXR_SEQUENCE)
CORE_IDX = np.asarray([r - 1 for r in PXR_CORE_POCKET_RESIDUES], dtype=np.int64)

CUTOFFS = (4.0, 6.0, 8.0, 10.0, 12.0)
SIGMAS = (2.0, 4.0, 6.0, 8.0)


def _protein_rep_coords(cif_path: str) -> np.ndarray:
    structure = gemmi.read_structure(cif_path)
    coords = np.full((PROTEIN_N_RES, 3), np.nan, dtype=np.float32)
    for model in structure:
        chain = model[PROTEIN_CHAIN_ID]
        for residue in chain:
            idx = int(residue.seqid.num) - 1
            if idx < 0 or idx >= PROTEIN_N_RES:
                continue
            fallback = None
            for atom in residue:
                if atom.element.name == "H":
                    continue
                coord = (atom.pos.x, atom.pos.y, atom.pos.z)
                if fallback is None:
                    fallback = coord
                if atom.name.strip() == "CA":
                    coords[idx] = coord
                    break
            if np.isnan(coords[idx, 0]) and fallback is not None:
                coords[idx] = fallback
        break
    return coords


def _ligand_coords(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as fh:
        mol = pickle.load(fh)
    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"Expected RDKit Mol in {pkl_path}, got {type(mol).__name__}")
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
    return np.asarray(coords, dtype=np.float32)


def _weighted_mean(z_block: np.ndarray, weights: np.ndarray) -> np.ndarray:
    flat_z = z_block.reshape(-1, z_block.shape[-1]).astype(np.float32)
    flat_w = weights.reshape(-1).astype(np.float32)
    total = float(flat_w.sum())
    if total <= 1e-8:
        return np.zeros(flat_z.shape[-1], dtype=np.float32)
    return (flat_z * flat_w[:, None]).sum(axis=0) / total


def _masked_max(z_block: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat_z = z_block.reshape(-1, z_block.shape[-1]).astype(np.float32)
    flat_m = mask.reshape(-1)
    if not flat_m.any():
        return np.zeros(flat_z.shape[-1], dtype=np.float32)
    return flat_z[flat_m].max(axis=0)


def _add_vec(row: dict[str, float], prefix: str, vec: np.ndarray) -> None:
    for i, value in enumerate(vec):
        row[f"{prefix}_{i:03d}"] = float(value)


def pool_one(task: tuple[int, str, str, str, int]) -> dict[str, float] | None:
    cid, npz_path, cif_path, pkl_path, ligand_atom_count = task
    try:
        data = np.load(npz_path, allow_pickle=False)
        z = data["z"][0].astype(np.float32)
        T = z.shape[0]
        lig_start = PROTEIN_N_RES
        lig_stop = min(T, PROTEIN_N_RES + int(ligand_atom_count))
        if lig_stop <= lig_start:
            return {"compound_id": int(cid), "_error": "empty ligand token span"}

        ligand_coords = _ligand_coords(pkl_path)
        n_lig = min(lig_stop - lig_start, ligand_coords.shape[0])
        if n_lig <= 0:
            return {"compound_id": int(cid), "_error": "empty ligand coords"}
        ligand_coords = ligand_coords[:n_lig]
        lig_stop = lig_start + n_lig

        protein_coords = _protein_rep_coords(cif_path)
        valid_res = np.isfinite(protein_coords).all(axis=1)
        diff = protein_coords[:, None, :] - ligand_coords[None, :, :]
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        dist[~valid_res, :] = np.inf

        z_x = z[:PROTEIN_N_RES, lig_start:lig_stop, :]
        row: dict[str, float] = {"compound_id": int(cid)}
        row["ligand_atom_count_used"] = float(n_lig)

        regions = {
            "all": np.where(valid_res)[0],
            "core": CORE_IDX[CORE_IDX < PROTEIN_N_RES],
        }
        for region_name, res_idx in regions.items():
            if res_idx.size == 0:
                continue
            z_region = z_x[res_idx, :, :]
            d_region = dist[res_idx, :]
            finite = np.isfinite(d_region)
            row[f"{region_name}_finite_pair_frac"] = float(finite.mean())
            row[f"{region_name}_min_dist"] = float(
                np.nanmin(np.where(finite, d_region, np.nan))
            )

            for cutoff in CUTOFFS:
                name = f"{cutoff:g}a".replace(".", "p")
                mask = np.isfinite(d_region) & (d_region <= cutoff)
                weights = mask.astype(np.float32)
                row[f"{region_name}_pair_frac_le_{name}"] = float(mask.mean())
                _add_vec(
                    row,
                    f"{region_name}_z_mean_le_{name}",
                    _weighted_mean(z_region, weights),
                )
                _add_vec(
                    row, f"{region_name}_z_max_le_{name}", _masked_max(z_region, mask)
                )

            for sigma in SIGMAS:
                name = f"{sigma:g}a".replace(".", "p")
                weights = np.exp(-0.5 * np.square(d_region / sigma)).astype(np.float32)
                weights[~np.isfinite(d_region)] = 0.0
                row[f"{region_name}_soft_weight_sum_{name}"] = float(weights.sum())
                _add_vec(
                    row,
                    f"{region_name}_z_softmean_s{name}",
                    _weighted_mean(z_region, weights),
                )

        return row
    except Exception as exc:  # noqa: BLE001
        return {"compound_id": int(cid), "_error": f"{type(exc).__name__}: {exc}"}


def _load_tasks() -> list[tuple[int, str, str, str, int]]:
    sql = """
    SELECT compound_id, embeddings_npz_path, pose_cif_path, ligand_pkl_path, ligand_atom_count
    FROM compound_boltz2
    WHERE embeddings_npz_path IS NOT NULL
      AND pose_cif_path IS NOT NULL
      AND ligand_pkl_path IS NOT NULL
      AND ligand_atom_count IS NOT NULL
      AND preprocessing_failed = FALSE
    ORDER BY compound_id
    """
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [
            (int(cid), str(npz), str(cif), str(pkl), int(n_lig))
            for cid, npz, cif, pkl, n_lig in cur.fetchall()
        ]


def _finalize(rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("compound_id").sort_index().astype(np.float32)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(
                float(df[col].mean()) if df[col].notna().any() else 0.0
            )
    nonconstant = [c for c in df.columns if df[c].nunique(dropna=False) > 1]
    return df[nonconstant]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = _load_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(f"Tasks: {len(tasks)}")

    rows: list[dict[str, float]] = []
    errors: list[dict[str, float]] = []
    if args.workers <= 1:
        iterator = map(pool_one, tasks)
        for row in tqdm(iterator, total=len(tasks), desc="dist-weighted-z"):
            if row is None:
                continue
            if "_error" in row:
                errors.append(row)
            else:
                rows.append(row)
    else:
        with Pool(args.workers) as pool:
            for row in tqdm(
                pool.imap_unordered(pool_one, tasks, chunksize=8),
                total=len(tasks),
                desc="dist-weighted-z",
            ):
                if row is None:
                    continue
                if "_error" in row:
                    errors.append(row)
                else:
                    rows.append(row)

    print(f"ok={len(rows)} errors={len(errors)}")
    if errors:
        print("First errors:")
        for err in errors[:5]:
            print(err)
    df = _finalize(rows)
    print(f"Feature frame: {df.shape}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="zstd")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
