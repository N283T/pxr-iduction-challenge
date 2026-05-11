"""Compute OpenEye FastROCS per-query score maps for selected prototypes.

The output parquet has one row per target compound and a JSON-compatible
``all_query_scores`` map: ``{query_compound_id: [shape, color, combo]}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS, load_train_smiles_target  # noqa: E402
from run_train import load_compound_ids  # noqa: E402

OMEGA_DIR = REPO_ROOT.joinpath("structures", "omega")


def sdf_path(compound_id: int) -> Path:
    return OMEGA_DIR.joinpath(f"{compound_id:05d}.sdf")


def load_ok_omega_ids(ids: list[int]) -> set[int]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        ph = ",".join(["%s"] * len(ids))
        df = pd.read_sql(
            f"SELECT compound_id FROM compound_omega3d WHERE status = 'ok' AND compound_id IN ({ph})",
            conn,
            params=ids,
        )
    return set(df["compound_id"].astype(int))


def select_queries(mode: str, n_queries: int) -> list[int]:
    train_df = load_train_smiles_target()
    train_ids = np.asarray(load_compound_ids("train"), dtype=int)
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    ok_ids = load_ok_omega_ids(train_ids.tolist())
    order = np.argsort(y if mode == "inactive" else -y, kind="stable")
    selected: list[int] = []
    for idx in order:
        cid = int(train_ids[idx])
        if cid in ok_ids and sdf_path(cid).exists():
            selected.append(cid)
        if len(selected) >= n_queries:
            break
    if not selected:
        raise ValueError(f"No {mode} queries selected")
    return selected


def build_shape_db(target_ids: list[int]) -> tuple[object, dict[int, int]]:
    from openeye import oechem, oefastrocs

    db = oefastrocs.OEShapeDatabase()
    idx_to_cid: dict[int, int] = {}
    for i, cid in enumerate(target_ids, start=1):
        path = sdf_path(cid)
        if not path.exists():
            continue
        ifs = oechem.oemolistream(str(path))
        mol = oechem.OEMol()
        while oechem.OEReadMolecule(ifs, mol):
            idx = db.AddMol(mol)
            idx_to_cid[idx] = cid
            mol = oechem.OEMol()
        ifs.close()
        if i % 1000 == 0:
            print(f"  loaded {i}/{len(target_ids)} targets ({len(idx_to_cid)} conformers)")
    return db, idx_to_cid


def query_scores(db, idx_to_cid: dict[int, int], query_id: int) -> dict[int, tuple[float, float, float]]:
    from openeye import oechem

    path = sdf_path(query_id)
    if not path.exists():
        return {}
    best: dict[int, tuple[float, float, float]] = {}
    ifs = oechem.oemolistream(str(path))
    query_mol = oechem.OEMol()
    while oechem.OEReadMolecule(ifs, query_mol):
        for result in db.GetSortedScores(query_mol):
            cid = idx_to_cid.get(result.GetMolIdx())
            if cid is None or cid == query_id:
                continue
            shape = float(result.GetShapeTanimoto())
            color = float(result.GetColorTanimoto())
            combo = float(result.GetTanimotoCombo())
            prev = best.get(cid)
            if prev is None or combo > prev[2]:
                best[cid] = (shape, color, combo)
        query_mol = oechem.OEMol()
    ifs.close()
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["active", "inactive"], default="inactive")
    parser.add_argument("--n-queries", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not os.environ.get("OE_LICENSE"):
        raise RuntimeError("OE_LICENSE env not set")

    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")
    target_ids_all = train_ids + test_ids
    ok_ids = load_ok_omega_ids(target_ids_all)
    target_ids = [cid for cid in target_ids_all if cid in ok_ids and sdf_path(cid).exists()]
    missing = len(target_ids_all) - len(target_ids)
    query_ids = select_queries(args.mode, args.n_queries)
    print(
        f"FastROCS query parquet: mode={args.mode} queries={len(query_ids)} "
        f"targets={len(target_ids)} missing_targets={missing}"
    )

    t0 = time.time()
    db, idx_to_cid = build_shape_db(target_ids)
    per_target: dict[int, dict[str, list[float]]] = {int(cid): {} for cid in target_ids_all}
    for i, qid in enumerate(query_ids, start=1):
        qt0 = time.time()
        scored = query_scores(db, idx_to_cid, qid)
        for target_id, values in scored.items():
            per_target[int(target_id)][str(qid)] = [round(v, 6) for v in values]
        print(
            f"  [{i}/{len(query_ids)}] query={qid} hits={len(scored)} "
            f"elapsed={time.time() - qt0:.1f}s"
        )

    df = pd.DataFrame(
        {
            "compound_id": list(per_target.keys()),
            "all_query_scores": [json.dumps(v, sort_keys=True) for v in per_target.values()],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {df.shape} to {args.output} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
