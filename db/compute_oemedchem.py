"""Compute OpenEye oemedchem + oemolprop 2D descriptors for every compound.

Outputs 16 scalar descriptors plus the Bemis-Murcko Framework SMILES into
``compound_oemedchem``.

Reads from ``compounds.std_smiles`` (ChEMBL-standardized SMILES). Runs
with multiprocessing workers since OE calls release the GIL poorly.

Requires ``OE_LICENSE=~/.openeye/oe_license.txt`` in the environment.
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
BATCH_SIZE = 500


def compute_one(task: tuple[int, str]) -> dict | None:
    """Compute OE descriptors for one compound. Returns dict or None on failure."""
    compound_id, smiles = task
    # Import inside worker so each process initializes OE cleanly.
    from openeye import oechem, oemedchem, oemolprop

    mol = oechem.OEGraphMol()
    if not oechem.OESmilesToMol(mol, smiles):
        return None
    oechem.OEAssignAromaticFlags(mol)

    try:
        xlogp = float(oemolprop.OEGetXLogP(mol))
    except Exception:
        xlogp = None
    try:
        psa_2d = float(oemolprop.OEGet2dPSA(mol))
    except Exception:
        psa_2d = None

    # Bemis-Murcko framework
    bm_framework = None
    try:
        for bm in oemedchem.OEGetBemisMurcko(mol):
            roles = [r.GetName() for r in bm.GetRoles()]
            if "Framework" in roles:
                scaf = oechem.OEGraphMol()
                oechem.OESubsetMol(scaf, mol, bm, True)
                bm_framework = oechem.OEMolToSmiles(scaf)
                break
    except Exception:
        pass

    return {
        "compound_id": compound_id,
        "xlogp": xlogp,
        "psa_2d": psa_2d,
        "mw": float(oechem.OECalculateMolecularWeight(mol)),
        "hba": int(oemolprop.OEGetHBondAcceptorCount(mol)),
        "hbd": int(oemolprop.OEGetHBondDonorCount(mol)),
        "lipinski_hba": int(oemolprop.OEGetLipinskiAcceptorCount(mol)),
        "lipinski_hbd": int(oemolprop.OEGetLipinskiDonorCount(mol)),
        "aromatic_ring_count": int(oemolprop.OEGetAromaticRingCount(mol)),
        "rotatable_bond_count": int(oemolprop.OEGetRotatableBondCount(mol)),
        "fraction_csp3": float(oemolprop.OEGetFractionCsp3(mol)),
        "halide_fraction": float(oemolprop.OEGetHalideFraction(mol)),
        "longest_unbranched_c_chain": int(
            oemolprop.OEGetLongestUnbranchedCarbonsChain(mol)
        ),
        "longest_unbranched_heavy_chain": int(
            oemolprop.OEGetLongestUnbranchedHeavyAtomsChain(mol)
        ),
        "anionic_carbon_count": int(oemolprop.OEGetAnionicCarbonCount(mol)),
        "num_unspecified_atom_stereo": int(
            oemolprop.OEGetNumUnspecifiedAtomStereos(mol)
        ),
        "num_unspecified_bond_stereo": int(
            oemolprop.OEGetNumUnspecifiedBondStereos(mol)
        ),
        "bemis_murcko_scaffold_smiles": bm_framework,
    }


def fetch_targets() -> list[tuple[int, str]]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.std_smiles
        FROM compounds c
        LEFT JOIN compound_oemedchem o ON o.compound_id = c.id
        WHERE o.compound_id IS NULL AND c.std_smiles IS NOT NULL
        ORDER BY c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


COLUMNS = (
    "compound_id",
    "xlogp",
    "psa_2d",
    "mw",
    "hba",
    "hbd",
    "lipinski_hba",
    "lipinski_hbd",
    "aromatic_ring_count",
    "rotatable_bond_count",
    "fraction_csp3",
    "halide_fraction",
    "longest_unbranched_c_chain",
    "longest_unbranched_heavy_chain",
    "anionic_carbon_count",
    "num_unspecified_atom_stereo",
    "num_unspecified_bond_stereo",
    "bemis_murcko_scaffold_smiles",
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
        INSERT INTO compound_oemedchem ({",".join(COLUMNS)})
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
    print(f"Computing oemedchem descriptors for {len(targets)} compounds...")
    t0 = time.time()

    done = 0
    failed = 0
    with Pool(processes=N_WORKERS) as pool:
        buf: list[dict] = []
        for result in pool.imap_unordered(compute_one, targets, chunksize=50):
            if result is None:
                failed += 1
                continue
            buf.append(result)
            done += 1
            if len(buf) >= BATCH_SIZE:
                upsert_batch(buf)
                buf = []
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(targets) - done) / rate if rate > 0 else 0
                print(
                    f"  done={done}/{len(targets)} "
                    f"failed={failed} rate={rate:.1f}/s eta={eta:.0f}s"
                )
        upsert_batch(buf)

    elapsed = time.time() - t0
    print(f"\nFinished: done={done} failed={failed} elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
