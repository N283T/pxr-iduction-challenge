#!/usr/bin/env -S pixi run python
"""Build compact Boltz-2 pose distogram-summary features.

This is a lightweight analogue of the Boltz-2 affinity head's distance
conditioning. The real head embeds a token-token distogram from the selected
predicted structure and feeds it through PairFormer. Here we keep only
low-dimensional summaries of protein-ligand atom distances so the feature can
be tested as an experimental add-on to pooled trunk embeddings.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "boltz2", "src", "boltz2"))
)

from constants import PXR_CORE_POCKET_RESIDUES, PROTEIN_CHAIN_ID  # noqa: E402
from data import DB_PARAMS  # noqa: E402

ATOM_OUT_PATH = REPO_ROOT.joinpath("data", "boltz2_distogram_features.parquet")
TOKEN_OUT_PATH = REPO_ROOT.joinpath("data", "boltz2_token_distogram_features.parquet")

DIST_BINS = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 22.0, np.inf])
BOLTZ_AFFINITY_BINS = np.concatenate(
    [
        np.asarray([0.0], dtype=np.float32),
        np.linspace(2.0, 22.0, 63, dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
    ]
)
CONTACT_CUTOFFS = (4.0, 6.0, 8.0, 10.0, 12.0)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SENTINEL_DISTANCE = 32.0


def _protein_residue_coords(cif_path: str) -> dict[int, np.ndarray]:
    structure = gemmi.read_structure(cif_path)
    residues: dict[int, list[tuple[float, float, float]]] = {}
    for model in structure:
        chain = model[PROTEIN_CHAIN_ID]
        for residue in chain:
            coords = residues.setdefault(int(residue.seqid.num), [])
            for atom in residue:
                if atom.element.name == "H":
                    continue
                coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
        break
    return {
        resno: np.asarray(coords, dtype=np.float32)
        for resno, coords in residues.items()
        if coords
    }


def _protein_residue_rep_coords(cif_path: str) -> dict[int, np.ndarray]:
    structure = gemmi.read_structure(cif_path)
    reps: dict[int, tuple[float, float, float]] = {}
    fallbacks: dict[int, tuple[float, float, float]] = {}
    for model in structure:
        chain = model[PROTEIN_CHAIN_ID]
        for residue in chain:
            resno = int(residue.seqid.num)
            for atom in residue:
                if atom.element.name == "H":
                    continue
                coord = (atom.pos.x, atom.pos.y, atom.pos.z)
                fallbacks.setdefault(resno, coord)
                if atom.name.strip() == "CA":
                    reps[resno] = coord
                    break
        break
    for resno, coord in fallbacks.items():
        reps.setdefault(resno, coord)
    return {resno: np.asarray(coord, dtype=np.float32) for resno, coord in reps.items()}


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


def summarize_distance_block(
    prefix: str,
    residue_coords_by_number: dict[int, np.ndarray],
    ligand_coords: np.ndarray,
    residue_numbers: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Summarize protein-ligand distances for a residue subset."""
    if residue_numbers is None:
        selected = residue_coords_by_number
    else:
        selected = {
            resno: residue_coords_by_number[resno]
            for resno in residue_numbers
            if resno in residue_coords_by_number
        }
    if not selected or ligand_coords.size == 0:
        return _missing_summary(prefix)

    residue_coords = np.concatenate(list(selected.values()), axis=0)
    dist = np.linalg.norm(
        residue_coords[:, None, :] - ligand_coords[None, :, :], axis=2
    ).astype(np.float32)
    flat = dist.reshape(-1)
    lig_min = dist.min(axis=0)

    residue_min = []
    for coords in selected.values():
        residue_dist = np.linalg.norm(
            coords[:, None, :] - ligand_coords[None, :, :], axis=2
        )
        residue_min.append(float(residue_dist.min()))
    residue_min_arr = np.asarray(residue_min, dtype=np.float32)

    out: dict[str, float] = {
        f"{prefix}_n_residues": float(len(selected)),
        f"{prefix}_n_residue_atoms": float(residue_coords.shape[0]),
        f"{prefix}_n_ligand_atoms": float(ligand_coords.shape[0]),
        f"{prefix}_pair_min": float(flat.min()),
        f"{prefix}_pair_mean": float(flat.mean()),
        f"{prefix}_pair_std": float(flat.std()),
        f"{prefix}_lig_min_mean": float(lig_min.mean()),
        f"{prefix}_lig_min_std": float(lig_min.std()),
        f"{prefix}_lig_min_max": float(lig_min.max()),
        f"{prefix}_res_min_mean": float(residue_min_arr.mean()),
        f"{prefix}_res_min_std": float(residue_min_arr.std()),
        f"{prefix}_res_min_max": float(residue_min_arr.max()),
    }

    for q in QUANTILES:
        q_name = f"q{int(q * 100):02d}"
        out[f"{prefix}_pair_{q_name}"] = float(np.quantile(flat, q))
        out[f"{prefix}_lig_min_{q_name}"] = float(np.quantile(lig_min, q))
        out[f"{prefix}_res_min_{q_name}"] = float(np.quantile(residue_min_arr, q))

    for cutoff in CONTACT_CUTOFFS:
        cut_name = f"{cutoff:g}a".replace(".", "p")
        out[f"{prefix}_pair_frac_le_{cut_name}"] = float((flat <= cutoff).mean())
        out[f"{prefix}_lig_atom_frac_le_{cut_name}"] = float((lig_min <= cutoff).mean())
        out[f"{prefix}_residue_frac_le_{cut_name}"] = float(
            (residue_min_arr <= cutoff).mean()
        )

    hist, _ = np.histogram(flat, bins=DIST_BINS)
    hist_frac = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    for i, frac in enumerate(hist_frac):
        lo = DIST_BINS[i]
        hi = DIST_BINS[i + 1]
        lo_name = f"{lo:g}".replace(".", "p")
        hi_name = "inf" if np.isinf(hi) else f"{hi:g}".replace(".", "p")
        out[f"{prefix}_pair_bin_{lo_name}_{hi_name}_frac"] = float(frac)

    return out


def summarize_token_distogram_block(
    prefix: str,
    residue_rep_coords_by_number: dict[int, np.ndarray],
    ligand_coords: np.ndarray,
    residue_numbers: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Summarize Boltz-affinity-style residue-token x ligand-token distances."""
    if residue_numbers is None:
        selected_items = sorted(residue_rep_coords_by_number.items())
    else:
        selected_items = [
            (resno, residue_rep_coords_by_number[resno])
            for resno in residue_numbers
            if resno in residue_rep_coords_by_number
        ]
    if not selected_items or ligand_coords.size == 0:
        return _missing_token_summary(prefix, residue_numbers)

    residue_numbers_present = [resno for resno, _ in selected_items]
    residue_coords = np.stack([coord for _, coord in selected_items]).astype(np.float32)
    dist = np.linalg.norm(
        residue_coords[:, None, :] - ligand_coords[None, :, :], axis=2
    ).astype(np.float32)
    flat = dist.reshape(-1)
    lig_min = dist.min(axis=0)
    res_min = dist.min(axis=1)

    hist, _ = np.histogram(flat, bins=BOLTZ_AFFINITY_BINS)
    hist_frac = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    finite_hi = BOLTZ_AFFINITY_BINS[1:]
    finite_hi = np.where(np.isinf(finite_hi), 22.0, finite_hi)
    expected_bin_hi = float(np.sum(hist_frac * finite_hi))
    nonzero = hist_frac[hist_frac > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())

    out: dict[str, float] = {
        f"{prefix}_token_n_residues": float(len(selected_items)),
        f"{prefix}_token_n_ligand_atoms": float(ligand_coords.shape[0]),
        f"{prefix}_token_pair_min": float(flat.min()),
        f"{prefix}_token_pair_mean": float(flat.mean()),
        f"{prefix}_token_pair_std": float(flat.std()),
        f"{prefix}_token_lig_min_mean": float(lig_min.mean()),
        f"{prefix}_token_res_min_mean": float(res_min.mean()),
        f"{prefix}_token_res_min_max": float(res_min.max()),
        f"{prefix}_token_boltz_hist_entropy": entropy,
        f"{prefix}_token_boltz_expected_bin_hi": expected_bin_hi,
    }

    for q in QUANTILES:
        q_name = f"q{int(q * 100):02d}"
        out[f"{prefix}_token_pair_{q_name}"] = float(np.quantile(flat, q))
        out[f"{prefix}_token_lig_min_{q_name}"] = float(np.quantile(lig_min, q))
        out[f"{prefix}_token_res_min_{q_name}"] = float(np.quantile(res_min, q))

    for cutoff in CONTACT_CUTOFFS:
        cut_name = f"{cutoff:g}a".replace(".", "p")
        out[f"{prefix}_token_pair_frac_le_{cut_name}"] = float((flat <= cutoff).mean())
        out[f"{prefix}_token_lig_atom_frac_le_{cut_name}"] = float(
            (lig_min <= cutoff).mean()
        )
        out[f"{prefix}_token_residue_frac_le_{cut_name}"] = float(
            (res_min <= cutoff).mean()
        )

    for i, frac in enumerate(hist_frac):
        out[f"{prefix}_token_boltz_bin_{i:02d}_frac"] = float(frac)

    if residue_numbers is not None:
        res_min_lookup = {
            resno: float(res_min[i]) for i, resno in enumerate(residue_numbers_present)
        }
        for resno in residue_numbers:
            min_dist = res_min_lookup.get(resno, SENTINEL_DISTANCE)
            out[f"{prefix}_res{resno}_token_min"] = min_dist
            for cutoff in (4.0, 6.0, 8.0):
                cut_name = f"{cutoff:g}a".replace(".", "p")
                out[f"{prefix}_res{resno}_token_contact_le_{cut_name}"] = float(
                    min_dist <= cutoff
                )

    return out


def _missing_token_summary(
    prefix: str, residue_numbers: tuple[int, ...] | None
) -> dict[str, float]:
    out = {
        f"{prefix}_token_n_residues": 0.0,
        f"{prefix}_token_n_ligand_atoms": 0.0,
        f"{prefix}_token_pair_min": SENTINEL_DISTANCE,
        f"{prefix}_token_pair_mean": SENTINEL_DISTANCE,
        f"{prefix}_token_pair_std": 0.0,
        f"{prefix}_token_lig_min_mean": SENTINEL_DISTANCE,
        f"{prefix}_token_res_min_mean": SENTINEL_DISTANCE,
        f"{prefix}_token_res_min_max": SENTINEL_DISTANCE,
        f"{prefix}_token_boltz_hist_entropy": 0.0,
        f"{prefix}_token_boltz_expected_bin_hi": SENTINEL_DISTANCE,
    }
    for q in QUANTILES:
        q_name = f"q{int(q * 100):02d}"
        out[f"{prefix}_token_pair_{q_name}"] = SENTINEL_DISTANCE
        out[f"{prefix}_token_lig_min_{q_name}"] = SENTINEL_DISTANCE
        out[f"{prefix}_token_res_min_{q_name}"] = SENTINEL_DISTANCE
    for cutoff in CONTACT_CUTOFFS:
        cut_name = f"{cutoff:g}a".replace(".", "p")
        out[f"{prefix}_token_pair_frac_le_{cut_name}"] = 0.0
        out[f"{prefix}_token_lig_atom_frac_le_{cut_name}"] = 0.0
        out[f"{prefix}_token_residue_frac_le_{cut_name}"] = 0.0
    for i in range(len(BOLTZ_AFFINITY_BINS) - 1):
        out[f"{prefix}_token_boltz_bin_{i:02d}_frac"] = 0.0
    if residue_numbers is not None:
        for resno in residue_numbers:
            out[f"{prefix}_res{resno}_token_min"] = SENTINEL_DISTANCE
            for cutoff in (4.0, 6.0, 8.0):
                cut_name = f"{cutoff:g}a".replace(".", "p")
                out[f"{prefix}_res{resno}_token_contact_le_{cut_name}"] = 0.0
    return out


def _missing_summary(prefix: str) -> dict[str, float]:
    out = {
        f"{prefix}_n_residues": 0.0,
        f"{prefix}_n_residue_atoms": 0.0,
        f"{prefix}_n_ligand_atoms": 0.0,
        f"{prefix}_pair_min": SENTINEL_DISTANCE,
        f"{prefix}_pair_mean": SENTINEL_DISTANCE,
        f"{prefix}_pair_std": 0.0,
        f"{prefix}_lig_min_mean": SENTINEL_DISTANCE,
        f"{prefix}_lig_min_std": 0.0,
        f"{prefix}_lig_min_max": SENTINEL_DISTANCE,
        f"{prefix}_res_min_mean": SENTINEL_DISTANCE,
        f"{prefix}_res_min_std": 0.0,
        f"{prefix}_res_min_max": SENTINEL_DISTANCE,
    }
    for q in QUANTILES:
        q_name = f"q{int(q * 100):02d}"
        out[f"{prefix}_pair_{q_name}"] = SENTINEL_DISTANCE
        out[f"{prefix}_lig_min_{q_name}"] = SENTINEL_DISTANCE
        out[f"{prefix}_res_min_{q_name}"] = SENTINEL_DISTANCE
    for cutoff in CONTACT_CUTOFFS:
        cut_name = f"{cutoff:g}a".replace(".", "p")
        out[f"{prefix}_pair_frac_le_{cut_name}"] = 0.0
        out[f"{prefix}_lig_atom_frac_le_{cut_name}"] = 0.0
        out[f"{prefix}_residue_frac_le_{cut_name}"] = 0.0
    for i in range(len(DIST_BINS) - 1):
        lo = DIST_BINS[i]
        hi = DIST_BINS[i + 1]
        lo_name = f"{lo:g}".replace(".", "p")
        hi_name = "inf" if np.isinf(hi) else f"{hi:g}".replace(".", "p")
        out[f"{prefix}_pair_bin_{lo_name}_{hi_name}_frac"] = 0.0
    return out


def extract_one(task: tuple[int, str, str, str]) -> dict[str, float] | None:
    cid, cif_path, pkl_path, mode = task
    try:
        ligand = _ligand_coords(pkl_path)
        row: dict[str, float] = {"compound_id": int(cid)}
        if mode == "atom":
            residues = _protein_residue_coords(cif_path)
            row.update(
                summarize_distance_block(
                    "core", residues, ligand, PXR_CORE_POCKET_RESIDUES
                )
            )
            row.update(summarize_distance_block("all", residues, ligand))
        elif mode == "token":
            reps = _protein_residue_rep_coords(cif_path)
            row.update(
                summarize_token_distogram_block(
                    "core", reps, ligand, PXR_CORE_POCKET_RESIDUES
                )
            )
            row.update(summarize_token_distogram_block("all", reps, ligand))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return row
    except Exception as exc:  # noqa: BLE001
        return {"compound_id": int(cid), "_error": str(exc)}


def _load_tasks() -> list[tuple[int, str, str]]:
    sql = """
    SELECT compound_id, pose_cif_path, ligand_pkl_path
    FROM compound_boltz2
    WHERE pose_cif_path IS NOT NULL
      AND ligand_pkl_path IS NOT NULL
    ORDER BY compound_id
    """
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [(int(cid), str(cif), str(pkl)) for cid, cif, pkl in cur.fetchall()]


def _finalize_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("compound_id").sort_index()
    df = df.astype(np.float32)
    nonconstant = [c for c in df.columns if df[c].nunique(dropna=False) > 1]
    return df[nonconstant]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("atom", "token"), default="atom")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = _load_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    out_path = args.out or (TOKEN_OUT_PATH if args.mode == "token" else ATOM_OUT_PATH)
    worker_tasks = [(cid, cif, pkl, args.mode) for cid, cif, pkl in tasks]
    print(f"Mode: {args.mode}")
    print(f"Tasks: {len(worker_tasks)}")

    rows: list[dict[str, float]] = []
    errors: list[dict[str, float]] = []
    if args.workers <= 1:
        iterator = map(extract_one, worker_tasks)
        for row in tqdm(iterator, total=len(worker_tasks), desc="distogram"):
            if row is None:
                continue
            if "_error" in row:
                errors.append(row)
            else:
                rows.append(row)
    else:
        with Pool(args.workers) as pool:
            for row in tqdm(
                pool.imap_unordered(extract_one, worker_tasks, chunksize=16),
                total=len(worker_tasks),
                desc="distogram",
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
    df = _finalize_frame(rows)
    print(f"Feature frame: {df.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
