"""Compute scikit-fingerprints 3D fingerprints from Boltz-2 pose pkls.

Two fingerprints that are NOT in vanilla RDKit and are not already
covered by the 984-dim RDKit 3D suite (tables compound_boltz2_desc3d
and compound_boltz2_desc3d_vector):

  E3FP (1024 bits) -- Axen et al. 2017 Morgan-style 3D fingerprint.
    Hashes 3D atomic environments instead of 2D subgraphs.
  ElectroShape (15 dim) -- USR generalised with partial charge as a
    5th dimension. Complements USR/USRCAT which are shape-only.

RUN ENVIRONMENT: this script must be run with the isolated uv venv
at /tmp/skfp-venv (not the project pixi env), because
scikit-fingerprints pins rdkit <= 2025.9.3 and the project uses
rdkit 2026.3.1. Boltz-2 pose pkls pickled with rdkit 2026.3.1
unpickle cleanly in the older rdkit (warning about version 16.3 ->
16.2 is benign; coords and bonds are preserved).

Usage:
    /tmp/skfp-venv/bin/python \\
      track1_activity/scripts/eda_cv_prep/10_compute_skfp_3d.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from rdkit import Chem
from skfp.fingerprints import (
    E3FPFingerprint,
    ElectroShapeFingerprint,
    PharmacophoreFingerprint,
)

warnings.filterwarnings("ignore")

# Ensure mol properties survive joblib's inter-process pickle.
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

DB_PARAMS = dict(host="/tmp", port=5433, dbname="pxr_challenge")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS compound_boltz2_skfp3d (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    e3fp JSONB,
    electroshape JSONB,
    pharmacophore3d JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE compound_boltz2_skfp3d
  ADD COLUMN IF NOT EXISTS pharmacophore3d JSONB;
"""

E3FP_BITS = 1024


def load_pose(pkl_path: str) -> Chem.Mol | None:
    try:
        with open(pkl_path, "rb") as f:
            m = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  pickle fail: {pkl_path} ({exc})")
        return None
    if not isinstance(m, Chem.Mol) or m.GetNumConformers() == 0:
        return None
    # skfp's require_mols_with_conf_ids expects the mol to carry this prop.
    m.SetIntProp("conf_id", int(m.GetConformer(0).GetId()))
    return m


def to_list_int(x) -> list:
    return np.asarray(x, dtype=np.int32).tolist()


def to_list_float(x) -> list:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, None).tolist()
    return arr


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    conn.commit()

    cur.execute(
        """
        SELECT compound_id, ligand_pkl_path
        FROM compound_boltz2
        WHERE ligand_pkl_path IS NOT NULL
        ORDER BY compound_id
        """
    )
    rows = cur.fetchall()
    print(f"Candidates with PKL: {len(rows)}")

    # Idempotent, but skip only rows that are FULLY populated (all 3
    # columns present). Adding a new FP later must re-run the missing
    # compounds transparently.
    cur.execute(
        """
        SELECT compound_id FROM compound_boltz2_skfp3d
        WHERE e3fp IS NOT NULL
          AND electroshape IS NOT NULL
          AND pharmacophore3d IS NOT NULL
        """
    )
    already = {r[0] for r in cur.fetchall()}
    todo = [(cid, p) for cid, p in rows if cid not in already]
    print(f"Already fully populated: {len(already)}, to compute: {len(todo)}")
    if not todo:
        return

    ids, mols = [], []
    t0 = time.time()
    for i, (cid, pkl) in enumerate(todo):
        mol = load_pose(pkl)
        if mol is None:
            continue
        ids.append(cid)
        mols.append(mol)
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(todo)} pkls in {time.time() - t0:.1f}s")
    print(f"Loaded {len(mols)} valid mols in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Running E3FP (parallel)...")
    fp_e3 = E3FPFingerprint(fp_size=E3FP_BITS, count=False, n_jobs=-1)
    e3fp_arr = fp_e3.transform(mols)
    print(f"  E3FP done: shape={e3fp_arr.shape} in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Running ElectroShape (parallel)...")
    fp_es = ElectroShapeFingerprint(
        partial_charge_model="formal",
        charge_errors="ignore",
        errors="ignore",
        n_jobs=-1,
    )
    es_arr = fp_es.transform(mols)
    print(f"  ElectroShape done: shape={es_arr.shape} in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("Running Pharmacophore3D (folded 2048, parallel)...")
    fp_ph = PharmacophoreFingerprint(
        variant="folded", fp_size=2048, use_3D=True, n_jobs=-1
    )
    ph_arr = fp_ph.transform(mols)
    print(f"  Pharmacophore3D done: shape={ph_arr.shape} in {time.time() - t2:.1f}s")

    print("Writing to DB...")
    batch = []
    for cid, e3_row, es_row, ph_row in zip(ids, e3fp_arr, es_arr, ph_arr):
        batch.append(
            (
                cid,
                json.dumps(to_list_int(e3_row)),
                json.dumps(to_list_float(es_row)),
                json.dumps(to_list_int(ph_row)),
            )
        )
    execute_batch(
        cur,
        """
        INSERT INTO compound_boltz2_skfp3d
            (compound_id, e3fp, electroshape, pharmacophore3d)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (compound_id) DO UPDATE SET
            e3fp = EXCLUDED.e3fp,
            electroshape = EXCLUDED.electroshape,
            pharmacophore3d = EXCLUDED.pharmacophore3d,
            computed_at = NOW()
        """,
        batch,
        page_size=200,
    )
    conn.commit()

    cur.execute(
        """
        SELECT COUNT(*), COUNT(e3fp), COUNT(electroshape), COUNT(pharmacophore3d)
        FROM compound_boltz2_skfp3d
        """
    )
    n, ne3, nes, nph = cur.fetchone()
    print(f"\nRows: total={n}  e3fp={ne3}  electroshape={nes}  pharmacophore3d={nph}")
    conn.close()


if __name__ == "__main__":
    main()
