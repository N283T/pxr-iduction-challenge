"""Compute OpenEye FastROCS shape similarity to the training potent-active set.

Queries: 67 training compounds with pEC50 >= 6.0 (multi-conformer from Omega).
Targets: all 13117 ``status='ok'`` compounds from compound_omega3d.
Scoring: FastROCS on GPU. Each SDF conformer is an independent DB entry;
         per (query_compound, target_compound) pair we take the max combo
         over all query-conf × target-conf overlays.

Output: ``compound_rocs`` table (one row per target compound).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OMEGA_DIR = REPO_ROOT.joinpath("structures", "omega")
BATCH_SIZE = 500
POTENT_PEC50 = 6.0


def sdf_path(compound_id: int) -> Path:
    return OMEGA_DIR.joinpath(f"{compound_id:05d}.sdf")


def load_query_compound_ids() -> list[int]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT t.compound_id
        FROM train_activity t
        JOIN compound_omega3d o ON o.compound_id = t.compound_id
        WHERE t.pec50 >= %s AND o.status = 'ok'
        ORDER BY t.compound_id
        """,
        (POTENT_PEC50,),
    )
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def load_target_compound_ids() -> list[int]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT compound_id FROM compound_omega3d
        WHERE status = 'ok' ORDER BY compound_id
        """
    )
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def build_shape_db_and_mapping(
    target_ids: list[int],
) -> tuple[object, dict[int, int]]:
    """Load all target conformers into a FastROCS DB. Returns (db, idx_to_cid)."""
    from openeye import oechem, oefastrocs

    db = oefastrocs.OEShapeDatabase()
    idx_to_cid: dict[int, int] = {}

    t0 = time.time()
    for i, cid in enumerate(target_ids):
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
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(
                f"  loaded {i + 1}/{len(target_ids)} compounds "
                f"({len(idx_to_cid)} confs) elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"  total: {len(idx_to_cid)} conformer entries loaded in {elapsed:.0f}s")
    return db, idx_to_cid


def query_one(
    db, query_id: int, idx_to_cid: dict[int, int]
) -> dict[int, tuple[float, float, float]]:
    """Return {target_cid: (shape, color, combo)} — max combo per target compound."""
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
            if cid is None:
                continue
            # Skip self-match: when target_id == query_id the overlay is
            # trivially perfect (combo ~2.0) and encodes "this compound is
            # pEC50 >= 6" directly (label leak L1). See issue #100 research
            # log for the leak analysis.
            # Env override ROCS_INCLUDE_SELF=1 kept only for deliberate
            # leak-verification LB experiments (2026-04-23 session).
            if cid == query_id and not os.environ.get("ROCS_INCLUDE_SELF"):
                continue
            shape = result.GetShapeTanimoto()
            color = result.GetColorTanimoto()
            combo = result.GetTanimotoCombo()
            prev = best.get(cid)
            if prev is None or combo > prev[2]:
                best[cid] = (shape, color, combo)
        query_mol = oechem.OEMol()
    ifs.close()
    return best


def summarize(
    per_query: dict[int, dict[int, tuple[float, float, float]]],
    target_ids: list[int],
) -> list[dict]:
    rows = []
    for tid in target_ids:
        all_scores: dict[int, tuple[float, float, float]] = {}
        for qid, scored in per_query.items():
            if tid in scored:
                all_scores[qid] = scored[tid]
        if not all_scores:
            continue

        shapes = [v[0] for v in all_scores.values()]
        colors = [v[1] for v in all_scores.values()]
        combos = [v[2] for v in all_scores.values()]
        best_qid, best_scores = max(all_scores.items(), key=lambda kv: kv[1][2])

        rows.append(
            {
                "compound_id": tid,
                "max_shape_tanimoto": max(shapes),
                "max_color_tanimoto": max(colors),
                "max_combo_tanimoto": max(combos),
                "mean_shape_tanimoto": sum(shapes) / len(shapes),
                "mean_color_tanimoto": sum(colors) / len(colors),
                "mean_combo_tanimoto": sum(combos) / len(combos),
                "nearest_query_compound_id": best_qid,
                "nearest_query_combo": best_scores[2],
                "all_query_scores": json.dumps(
                    {
                        str(qid): [round(s, 4) for s in v]
                        for qid, v in all_scores.items()
                    }
                ),
            }
        )
    return rows


COLUMNS = (
    "compound_id",
    "max_shape_tanimoto",
    "max_color_tanimoto",
    "max_combo_tanimoto",
    "mean_shape_tanimoto",
    "mean_color_tanimoto",
    "mean_combo_tanimoto",
    "nearest_query_compound_id",
    "nearest_query_combo",
    "all_query_scores",
)


def upsert(rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    values = [tuple(r[c] for c in COLUMNS) for r in rows]
    execute_values(
        cur,
        f"""
        INSERT INTO compound_rocs ({",".join(COLUMNS)})
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

    print("Loading query / target compound ids...")
    query_ids = load_query_compound_ids()
    target_ids = load_target_compound_ids()
    print(f"  {len(query_ids)} queries, {len(target_ids)} targets")

    t0 = time.time()
    print("\nBuilding FastROCS shape database...")
    db, idx_to_cid = build_shape_db_and_mapping(target_ids)

    print(f"\nRunning FastROCS for {len(query_ids)} queries...")
    per_query: dict[int, dict[int, tuple[float, float, float]]] = {}
    for i, qid in enumerate(query_ids, start=1):
        qt0 = time.time()
        scored = query_one(db, qid, idx_to_cid)
        per_query[qid] = scored
        elapsed = time.time() - qt0
        print(
            f"  [{i}/{len(query_ids)}] query={qid:>5d} hits={len(scored)} "
            f"elapsed={elapsed:.1f}s"
        )

    print("\nAggregating per-compound summaries...")
    rows = summarize(per_query, target_ids)
    print(f"  {len(rows)} compounds with >= 1 scored query")

    print("Upserting to compound_rocs...")
    for start in range(0, len(rows), BATCH_SIZE):
        upsert(rows[start : start + BATCH_SIZE])

    total_elapsed = time.time() - t0
    print(f"\nFinished in {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
