"""Fix compounds with geometrically impossible stereochemistry.

Some SMILES in the dataset specify bridgehead CIP configurations that cannot
be realized in 3D (trans-bridgeheads in a bicyclo[2.2.1] framework, etc.).
ETKDGv3 correctly refuses to embed them, which breaks any pipeline that
builds 3D coordinates downstream (Boltz-2, conformer descriptors, etc.).

This script scans all compounds, detects the impossible-stereo cases, and
replaces each one with the stereoisomer closest to the original (by Hamming
distance over CIP labels) that *can* be embedded. Ties are broken by the
lex-smallest canonical SMILES so the output is deterministic.

An audit CSV is written to db/bridged_stereo_fixes.csv documenting every
change (original SMILES, fixed SMILES, CIP before/after, Hamming distance).

Idempotent: re-running is a no-op once the DB is already fixed.

Usage:
    pixi run python db/fix_bridged_stereo.py              # dry-run (default)
    pixi run python db/fix_bridged_stereo.py --apply      # commit DB updates
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT.joinpath("db", "bridged_stereo_fixes.csv")

EMBED_SEEDS = (42, 1, 7)


def try_embed(mol: Chem.Mol) -> bool:
    """Return True if ETKDGv3 succeeds on any seed from EMBED_SEEDS."""
    mh = Chem.AddHs(mol)
    for seed in EMBED_SEEDS:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, params) == 0:
            return True
    return False


def get_cip(mol: Chem.Mol) -> tuple[tuple[int, str], ...]:
    """Return tuple of (atom_idx, CIP code) for CIP-labeled atoms, sorted."""
    m = Chem.Mol(mol)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return tuple(
        sorted(
            (a.GetIdx(), a.GetPropsAsDict().get("_CIPCode"))
            for a in m.GetAtoms()
            if a.HasProp("_CIPCode")
        )
    )


def hamming_cip(
    cip_a: tuple[tuple[int, str], ...],
    cip_b: tuple[tuple[int, str], ...],
) -> int | None:
    """Hamming distance over CIP codes; None if atom-sets differ."""
    if tuple(a[0] for a in cip_a) != tuple(b[0] for b in cip_b):
        return None
    return sum(1 for a, b in zip(cip_a, cip_b) if a[1] != b[1])


def find_repair(
    smiles: str,
) -> tuple[str, tuple, tuple, int] | None:
    """Return (fixed_smiles, orig_cip, fixed_cip, hamming_distance) or None.

    None is returned when the SMILES is already embeddable (no repair needed)
    or when no embeddable stereoisomer can be found.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if try_embed(mol):
        return None
    orig_cip = get_cip(mol)
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=True)
    candidates: list[tuple[int, str, tuple]] = []
    for iso in EnumerateStereoisomers(mol, options=opts):
        if not try_embed(iso):
            continue
        iso_cip = get_cip(iso)
        h = hamming_cip(orig_cip, iso_cip)
        if h is None:
            continue
        candidates.append((h, Chem.MolToSmiles(iso), iso_cip))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    h, fixed_smi, fixed_cip = candidates[0]
    return fixed_smi, orig_cip, fixed_cip, h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit DB updates (default: dry-run only)",
    )
    parser.add_argument("--host", default="/tmp")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--dbname", default="pxr_challenge")
    args = parser.parse_args()

    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(
        "SELECT id, std_smiles FROM compounds "
        "WHERE std_smiles IS NOT NULL ORDER BY id"
    )
    rows = cur.fetchall()
    print(f"Scanning {len(rows)} compounds...")

    fixes: list[dict] = []
    for i, (cid, smi) in enumerate(rows):
        result = find_repair(smi)
        if result is not None:
            fixed_smi, orig_cip, fixed_cip, h = result
            fixes.append(
                {
                    "compound_id": cid,
                    "original_smiles": smi,
                    "fixed_smiles": fixed_smi,
                    "original_cip": str(orig_cip),
                    "fixed_cip": str(fixed_cip),
                    "hamming_distance": h,
                }
            )
            print(f"  id={cid}  h={h}  {smi}  ->  {fixed_smi}")
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i+1}/{len(rows)}")

    print(f"\nRepair candidates: {len(fixes)}")
    if not fixes:
        print("No fixes needed; DB already consistent.")
        return

    fixes.sort(key=lambda x: x["compound_id"])
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "compound_id",
                "original_smiles",
                "fixed_smiles",
                "original_cip",
                "fixed_cip",
                "hamming_distance",
                "fix_timestamp",
            ],
        )
        writer.writeheader()
        for fx in fixes:
            writer.writerow({**fx, "fix_timestamp": timestamp})
    print(f"Audit CSV -> {CSV_PATH}")

    if not args.apply:
        print("\n[DRY-RUN] DB not modified. Pass --apply to commit changes.")
        return

    for fx in fixes:
        cur.execute(
            "UPDATE compounds "
            "SET std_smiles = %s, std_mol = mol_from_smiles(%s::cstring) "
            "WHERE id = %s",
            (fx["fixed_smiles"], fx["fixed_smiles"], fx["compound_id"]),
        )
    conn.commit()
    print(f"Applied {len(fixes)} DB updates (committed).")


if __name__ == "__main__":
    main()
