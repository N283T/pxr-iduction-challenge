#!/usr/bin/env -S pixi run python
"""Score Track 2 candidate poses by overlap with FTMap consensus hotspots.

FTMap was run on our PXR LBD apo prediction (uploaded to the FTMap web
server) and returned a multi-chain PDB at
``structures/ftmap/pxr_apo_ftmap.pdb``:

  - chain A: the apo PXR LBD protein
  - chains P, Q, R, ..., Z: consensus probe clusters ("CSs"), each a
    spatial cluster of 16 different probe types (cyclohexane, phenol,
    urea, acetonitrile, ...). FTMap convention: alphabetical order
    encodes RANK (Z = top, P = lowest), measured by total probe atoms
    in the cluster.

Top consensus sites for our PXR run (atoms / unique probes):
  Z 94 / 16    — canonical pocket (all 16 probes converge here)
  Y 68 / 13    — secondary
  X 64 / 11
  W 63 /  8

This scorer:
  1. Loads the FTMap PDB. Reads chain A Cα + each consensus chain's
     atoms.
  2. Per query: Cα-superposes the FTMap apo onto the query's Boltz
     ``model_0`` protein (apo and holo are near-identical at Cα level
     for the same LBD sequence) → (R, t).
  3. Transforms each consensus site centroid into the Boltz frame.
  4. For each candidate pose source (Boltz model_0..4, gnina-refined,
     template_transferred, posit), computes:
       - Euclidean distance from pose centroid to top hotspot Z
       - Distance to the nearest hotspot among the top 4 (W/X/Y/Z)
       - "Hotspot enrichment" = how many of the top-4 sites the pose
         centroid is within 5 Å of (0-4)

Output: ``docs/track2/track2_ftmap_hotspot_scores.csv`` with one row per
(query, source) — joins to the existing ``track2_template_rmsd_scores.csv``
on ``compound`` + ``source``.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("OE_LICENSE", os.path.expanduser("~/.openeye/oe_license.txt"))

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from track2.constants import REPO_ROOT as PROJECT_ROOT  # noqa: E402

PRED_DIR = PROJECT_ROOT.joinpath(
    "structures",
    "boltz2_track2",
    "outputs",
    "holo",
    "boltz_results_holo",
    "predictions",
)
GNINA_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_gnina")
TEMPLATE_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_template")
POSIT_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_posit")
FTMAP_PDB = PROJECT_ROOT.joinpath("structures", "ftmap", "pxr_apo_ftmap.pdb")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")
OUT_CSV = PROJECT_ROOT.joinpath("docs", "track2", "track2_ftmap_hotspot_scores.csv")


# ---------------------------------------------------------------------------
# FTMap parsing
# ---------------------------------------------------------------------------


def _read_ftmap(pdb_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (chain A Cα xyz, chain A residue numbers, consensus dict).

    Consensus dict maps chain id (P, Q, ..., Z) to (N, 3) atom array.
    """
    a_xyz: list[tuple[float, float, float]] = []
    a_res: list[int] = []
    cons: dict[str, list[tuple[float, float, float]]] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            ch = line[21]
            atom_name = line[12:16].strip()
            resnum = int(line[22:26].strip())
        except ValueError:
            continue
        if ch == "A":
            if atom_name == "CA":
                a_xyz.append((x, y, z))
                a_res.append(resnum)
        else:
            cons.setdefault(ch, []).append((x, y, z))
    return (
        np.asarray(a_xyz),
        np.asarray(a_res, dtype=int),
        {ch: np.asarray(xyz) for ch, xyz in cons.items()},
    )


def _read_pdb_ca(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[tuple[float, float, float]] = []
    resids: list[int] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            r = int(line[22:26].strip())
        except ValueError:
            continue
        xyz.append((x, y, z))
        resids.append(r)
    return np.asarray(xyz), np.asarray(resids, dtype=int)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cp = P.mean(0)
    cq = Q.mean(0)
    Pc = P - cp
    Qc = Q - cq
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    aligned = (R @ Pc.T).T + cq
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean()))
    return R, t, rmsd


def _ligand_centroid_from_pdb(pdb_path: Path) -> np.ndarray | None:
    xyz: list[tuple[float, float, float]] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("HETATM") or len(line) < 54:
            continue
        if line[17:20].strip() != "LIG":
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        xyz.append((x, y, z))
    return np.asarray(xyz).mean(0) if xyz else None


def _process_one(
    qid: str,
    ftmap_a_ca: np.ndarray,
    ftmap_a_res: np.ndarray,
    cons_chains: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    boltz_dir = PRED_DIR.joinpath(qid)
    model0 = boltz_dir.joinpath(f"{qid}_model_0.pdb")
    if not model0.exists():
        return [{"compound": qid, "error": "boltz_pdb_missing"}]
    boltz_ca, boltz_res = _read_pdb_ca(model0)

    # Match by residue number (FTMap apo is on the LBD sequence too,
    # numbered 1..293 same as Boltz).
    p_lookup = {int(r): i for i, r in enumerate(ftmap_a_res)}
    q_lookup = {int(r): i for i, r in enumerate(boltz_res)}
    common = sorted(set(p_lookup) & set(q_lookup))
    if len(common) < 50:
        n = min(len(ftmap_a_ca), len(boltz_ca))
        P = ftmap_a_ca[:n]
        Q = boltz_ca[:n]
    else:
        P = np.stack([ftmap_a_ca[p_lookup[r]] for r in common])
        Q = np.stack([boltz_ca[q_lookup[r]] for r in common])
    R, t, ca_rmsd = _kabsch(P, Q)

    # Apply (R, t) to consensus site centroids
    centroids: dict[str, np.ndarray] = {}
    for ch, atoms in cons_chains.items():
        centroid_ftmap = atoms.mean(0)
        centroids[ch] = R @ centroid_ftmap + t

    # Top-4 hotspots in our run: Z, Y, X, W (largest clusters)
    top4 = ["Z", "Y", "X", "W"]
    top4_xyz = np.stack([centroids[ch] for ch in top4 if ch in centroids])

    # Score every candidate pose source
    candidate_pdbs: dict[str, Path] = {
        f"boltz_model_{i}": boltz_dir.joinpath(f"{qid}_model_{i}.pdb") for i in range(5)
    }
    candidate_pdbs["gnina_refined"] = GNINA_DIR.joinpath(qid, f"{qid}_refined.pdb")
    candidate_pdbs["template_transferred"] = TEMPLATE_DIR.joinpath(
        qid, f"{qid}_template.pdb"
    )
    candidate_pdbs["posit"] = POSIT_DIR.joinpath(qid, f"{qid}_posit.pdb")

    base = {"compound": qid, "ftmap_ca_rmsd": ca_rmsd}
    for ch in top4:
        if ch in centroids:
            xyz = centroids[ch]
            base[f"hotspot_{ch}_xyz"] = f"{xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}"

    rows = []
    for source, pdb_path in candidate_pdbs.items():
        row = {**base, "source": source}
        if not pdb_path.exists():
            row["error"] = "pdb_missing"
            rows.append(row)
            continue

        # For Boltz models 1..4: also Cα-superpose to model_0
        # (each Boltz output is in its own diffusion-sample frame).
        if source.startswith("boltz_model_") and source != "boltz_model_0":
            this_ca, this_res = _read_pdb_ca(pdb_path)
            common2 = sorted(set(this_res) & set(boltz_res))
            if len(common2) >= 50:
                lookup_a = {int(r): i for i, r in enumerate(this_res)}
                lookup_b = {int(r): i for i, r in enumerate(boltz_res)}
                Pa = np.stack([this_ca[lookup_a[r]] for r in common2])
                Qb = np.stack([boltz_ca[lookup_b[r]] for r in common2])
                Rm, tm, _ = _kabsch(Pa, Qb)
            else:
                Rm, tm = np.eye(3), np.zeros(3)
        else:
            Rm, tm = np.eye(3), np.zeros(3)

        cent = _ligand_centroid_from_pdb(pdb_path)
        if cent is None:
            row["error"] = "no_ligand_atoms"
            rows.append(row)
            continue
        cent_in_boltz = Rm @ cent + tm
        row["ligand_centroid"] = (
            f"{cent_in_boltz[0]:.2f},{cent_in_boltz[1]:.2f},{cent_in_boltz[2]:.2f}"
        )

        # Distance to top hotspot Z
        if "Z" in centroids:
            row["dist_to_Z"] = float(np.linalg.norm(cent_in_boltz - centroids["Z"]))
        # Distance to nearest of top-4
        dists = np.linalg.norm(top4_xyz - cent_in_boltz, axis=1)
        row["dist_to_nearest_top4"] = float(dists.min())
        row["nearest_top4_chain"] = top4[int(dists.argmin())]
        # Hotspot enrichment: # of top-4 sites within 5 Å
        row["n_top4_within_5A"] = int((dists < 5.0).sum())
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ftmap", type=Path, default=FTMAP_PDB)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    if not args.ftmap.exists():
        sys.exit(f"FTMap PDB not found: {args.ftmap}")

    print(f"Reading {args.ftmap}")
    ft_a_ca, ft_a_res, cons = _read_ftmap(args.ftmap)
    print(
        f"FTMap apo Cα: {len(ft_a_ca)}, residue range: {ft_a_res.min()}-{ft_a_res.max()}"
    )
    print(f"Consensus sites ({len(cons)}):")
    for ch in sorted(cons.keys()):
        n = len(cons[ch])
        cent = cons[ch].mean(0)
        print(f"  {ch}: {n} atoms, centroid {cent.round(2)}")

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    print(f"\nQueries: {len(df)}")

    all_rows: list[dict[str, Any]] = []
    for n, (_, row) in enumerate(df.iterrows(), 1):
        all_rows.extend(_process_one(row.structure, ft_a_ca, ft_a_res, cons))
        if n % 25 == 0 or n == len(df):
            print(f"  {n}/{len(df)}")

    out_df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out_df)} rows)")

    scored = (
        out_df[out_df.get("dist_to_Z").notna()]
        if "dist_to_Z" in out_df.columns
        else pd.DataFrame()
    )
    if not scored.empty:
        print("\n=== Distance to top hotspot Z, by source (lower = better) ===")
        print(
            scored.groupby("source")[["dist_to_Z", "dist_to_nearest_top4"]]
            .agg(["mean", "median"])
            .round(2)
            .to_string()
        )
        print("\n=== Top-4 hotspots within 5 Å count, by source (higher = better) ===")
        print(
            scored.groupby("source")["n_top4_within_5A"]
            .agg(["mean", "median", "max"])
            .round(2)
            .to_string()
        )


if __name__ == "__main__":
    main()
