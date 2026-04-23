"""Compute OpenEye Omega 3D conformer ensembles.

Input: pH 7.4 protonated SMILES from ``compound_quacpac.ph74_smiles``.
Output: one SDF per compound under ``structures/omega/<id>.sdf``
(multi-conformer, max 10 confs, 10 kcal/mol energy window).

Used downstream by ROCS shape similarity and Uni-Mol Path 2 encoding.
Requires ``OE_LICENSE=~/.openeye/oe_license.txt``.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

N_WORKERS = 8
BATCH_SIZE = 200
OMEGA_DIR = REPO_ROOT.joinpath("structures", "omega")
OMEGA_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONFS = 10
ENERGY_WINDOW = 10.0  # kcal/mol
RMS_THRESHOLD = 0.5  # angstrom
MAX_HEAVY_ATOMS = 80  # guardrail — Omega struggles on very large peptides/macrolides


def compute_one(task: tuple[int, str]) -> dict:
    compound_id, smiles = task
    sdf_path = OMEGA_DIR.joinpath(f"{compound_id:05d}.sdf")
    result = {
        "compound_id": compound_id,
        "input_smiles": smiles,
        "sdf_path": str(sdf_path),
        "n_confs": 0,
        "min_energy": None,
        "max_energy": None,
        "status": None,
    }
    from openeye import oechem, oeomega

    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        result["status"] = "parse_failed"
        return result

    # Guardrail: skip very large molecules
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy > MAX_HEAVY_ATOMS:
        result["status"] = "too_large"
        return result

    # Configure Omega
    opts = oeomega.OEOmegaOptions()
    opts.SetMaxConfs(MAX_CONFS)
    opts.SetEnergyWindow(ENERGY_WINDOW)
    opts.SetRMSThreshold(RMS_THRESHOLD)
    opts.SetStrictStereo(False)  # allow unspecified stereo to flip
    opts.SetWarts(False)
    omega = oeomega.OEOmega(opts)

    if not omega(mol):
        result["status"] = "omega_failed"
        return result

    # Collect energies
    energies = []
    for conf in mol.GetConfs():
        # Omega stores SD data with energy on the conformer
        try:
            e = float(oechem.OEGetSDData(conf, "Energy"))
            energies.append(e)
        except Exception:
            pass

    # Write SDF
    ofs = oechem.oemolostream(str(sdf_path))
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()

    result["n_confs"] = mol.NumConfs()
    if energies:
        result["min_energy"] = min(energies)
        result["max_energy"] = max(energies)
    result["status"] = "ok"
    return result


def fetch_targets() -> list[tuple[int, str]]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    # Use pH 7.4 SMILES (falls back to std_smiles if missing)
    cur.execute(
        """
        SELECT c.id, COALESCE(q.ph74_smiles, c.std_smiles) AS smi
        FROM compounds c
        LEFT JOIN compound_quacpac q ON q.compound_id = c.id
        LEFT JOIN compound_omega3d o ON o.compound_id = c.id
        WHERE o.compound_id IS NULL
          AND COALESCE(q.ph74_smiles, c.std_smiles) IS NOT NULL
        ORDER BY c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


COLUMNS = (
    "compound_id",
    "input_smiles",
    "sdf_path",
    "n_confs",
    "min_energy",
    "max_energy",
    "status",
)


def upsert_batch(rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    values = [tuple(r[c] for c in COLUMNS) for r in rows]
    execute_values(
        cur,
        f"""
        INSERT INTO compound_omega3d ({",".join(COLUMNS)})
        VALUES %s
        ON CONFLICT (compound_id) DO UPDATE SET
          {",".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "compound_id")},
          computed_at = now()
        """,
        values,
    )
    conn.commit()
    cur.close()
    conn.close()


def main() -> None:
    if not os.environ.get("OE_LICENSE"):
        raise RuntimeError(
            "OE_LICENSE env not set. Run with "
            "OE_LICENSE=~/.openeye/oe_license.txt pixi run python ..."
        )

    targets = fetch_targets()
    print(f"Generating Omega 3D conformers for {len(targets)} compounds...")
    print(
        f"  max_confs={MAX_CONFS} ewindow={ENERGY_WINDOW} "
        f"rms={RMS_THRESHOLD} max_heavy={MAX_HEAVY_ATOMS}"
    )
    t0 = time.time()

    counts: dict[str, int] = {}
    with Pool(processes=N_WORKERS) as pool:
        buf: list[dict] = []
        for i, result in enumerate(
            pool.imap_unordered(compute_one, targets, chunksize=20), start=1
        ):
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            buf.append(result)
            if len(buf) >= BATCH_SIZE:
                upsert_batch(buf)
                buf = []
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(targets) - i) / rate if rate > 0 else 0
                status_str = " ".join(f"{k}={v}" for k, v in counts.items())
                print(
                    f"  processed={i}/{len(targets)} rate={rate:.1f}/s "
                    f"eta={eta:.0f}s | {status_str}"
                )
        upsert_batch(buf)

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
