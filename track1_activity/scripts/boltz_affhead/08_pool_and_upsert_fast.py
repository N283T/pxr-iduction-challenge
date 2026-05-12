"""Pool Boltz-2 trunk s/z embeddings to a 1024-dim vector and upsert
into compound_boltz2_trunk_fast.

Source sets:
  historical  --  rcycle=3 npz from existing compound_boltz2.embeddings_npz_path
                  (4,653 train+test+counter compounds, finalized)
  fast        --  rcycle=1 npz from structures/boltz2/outputs_fast/.../predictions/<id>/
                  (8,482 missing compounds, populated by boltz2_fast_embeddings_run.sh)
  fast-r3     --  rcycle=3 npz from structures/boltz2/outputs_fast_rcycle3/.../predictions/<id>/
                  (same fast-only compounds, upgraded by resuming from rcycle=1)

Pool variant: all_pairs (Boltz-2 paper's interface mask, minus lig x lig
diagonal). Matches 01b_pool_allpairs.py (same 1024d schema).

Safe to run while the fast Boltz job is still writing: files modified within
--min-age-seconds are skipped, and BadZipFile errors are swallowed (file
truncated mid-write). Re-run after Boltz completes to catch the latest.

Usage
-----
    # Both sources, skip files <60s old (default)
    pixi run python track1_activity/scripts/boltz_affhead/08_pool_and_upsert_fast.py

    # Only the historical (rcycle=3) set
    pixi run python track1_activity/scripts/boltz_affhead/08_pool_and_upsert_fast.py --source historical

    # Only the fast (rcycle=1) set, force re-pool everything
    pixi run python track1_activity/scripts/boltz_affhead/08_pool_and_upsert_fast.py --source fast --force

    # Upsert upgraded rcycle=3 fast compounds as they finish
    pixi run python track1_activity/scripts/boltz_affhead/08_pool_and_upsert_fast.py --source fast-r3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "boltz2", "src", "boltz2"))
)

from constants import PXR_SEQUENCE  # noqa: E402
from data import DB_PARAMS  # noqa: E402

PROTEIN_N_RES = len(PXR_SEQUENCE)  # 434

FAST_PRED_DIR = REPO_ROOT.joinpath(
    "structures",
    "boltz2",
    "outputs_fast",
    "boltz_results_inputs_fast",
    "predictions",
)
FAST_R3_PRED_DIR = REPO_ROOT.joinpath(
    "structures",
    "boltz2",
    "outputs_fast_rcycle3",
    "boltz_results_inputs_fast",
    "predictions",
)


def pool_npz(npz_path: Path) -> dict[str, list[float]] | None:
    """Pool one embeddings_*.npz to 4 vectors (1024 dims total).

    Returns None if the file is unreadable / truncated / shape-invalid.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
        s = data["s"][0]  # (T, 384)
        z = data["z"][0]  # (T, T, 128)
    except Exception as e:  # noqa: BLE001
        print(f"  skip {npz_path.name}: load error {type(e).__name__}: {e}")
        return None

    T = s.shape[0]
    if T <= PROTEIN_N_RES:
        print(f"  skip {npz_path.name}: T={T} <= PROTEIN_N_RES={PROTEIN_N_RES}")
        return None

    prot_slice = slice(0, PROTEIN_N_RES)
    lig_slice = slice(PROTEIN_N_RES, T)

    s_prot = s[prot_slice].mean(axis=0).astype(np.float32)
    s_lig = s[lig_slice].mean(axis=0).astype(np.float32)

    z_xp = z[prot_slice, lig_slice, :]  # (PROTEIN_N_RES, L, 128)
    if z_xp.size == 0:
        return None
    z_xp_flat = z_xp.reshape(-1, z_xp.shape[-1])
    z_if_mean = z_xp_flat.mean(axis=0).astype(np.float32)
    z_if_max = z_xp_flat.max(axis=0).astype(np.float32)

    return dict(
        s_prot_mean=s_prot.tolist(),
        s_lig_mean=s_lig.tolist(),
        z_if_mean=z_if_mean.tolist(),
        z_if_max=z_if_max.tolist(),
    )


def iter_historical(conn) -> Iterable[tuple[int, Path]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT compound_id, embeddings_npz_path
            FROM compound_boltz2
            WHERE embeddings_npz_path IS NOT NULL
              AND preprocessing_failed = FALSE
            ORDER BY compound_id
            """
        )
        for cid, path in cur.fetchall():
            yield int(cid), Path(path)


def iter_prediction_dir(pred_dir: Path) -> Iterable[tuple[int, Path]]:
    if not pred_dir.exists():
        return
    for cid_dir in sorted(pred_dir.iterdir()):
        if not cid_dir.is_dir():
            continue
        try:
            cid = int(cid_dir.name)
        except ValueError:
            continue
        npz = cid_dir.joinpath(f"embeddings_{cid_dir.name}.npz")
        if npz.exists():
            yield cid, npz


def iter_fast() -> Iterable[tuple[int, Path]]:
    yield from iter_prediction_dir(FAST_PRED_DIR)


def iter_fast_r3() -> Iterable[tuple[int, Path]]:
    yield from iter_prediction_dir(FAST_R3_PRED_DIR)


def existing_ids(conn, recycling_steps: int) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT compound_id FROM compound_boltz2_trunk_fast WHERE recycling_steps = %s",
            (recycling_steps,),
        )
        return {row[0] for row in cur.fetchall()}


def upsert_batch(conn, batch: list[tuple]) -> None:
    """Upsert a batch of pooled rows.

    Each row tuple: (cid, s_prot, s_lig, z_mean, z_max, rcycle, src_path)
    """
    if not batch:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO compound_boltz2_trunk_fast
                (compound_id, s_prot_mean, s_lig_mean, z_if_mean, z_if_max,
                 recycling_steps, source_npz_path)
            VALUES %s
            ON CONFLICT (compound_id) DO UPDATE SET
                s_prot_mean = EXCLUDED.s_prot_mean,
                s_lig_mean = EXCLUDED.s_lig_mean,
                z_if_mean = EXCLUDED.z_if_mean,
                z_if_max = EXCLUDED.z_if_max,
                recycling_steps = EXCLUDED.recycling_steps,
                source_npz_path = EXCLUDED.source_npz_path,
                created_at = NOW()
            """,
            batch,
        )
    conn.commit()


def process_source(
    conn,
    source: str,
    iterator: Iterable[tuple[int, Path]],
    rcycle: int,
    force: bool,
    min_age_seconds: int,
    batch_size: int,
) -> tuple[int, int, int]:
    """Returns (n_processed, n_skipped_existing, n_skipped_other)."""
    skip_ids = set() if force else existing_ids(conn, rcycle)
    if not force:
        print(f"[{source}] {len(skip_ids)} compounds already in DB at rcycle={rcycle}")

    now = time.time()
    batch: list[tuple] = []
    n_done, n_skip_existing, n_skip_other = 0, 0, 0
    seen = 0

    for cid, npz in iterator:
        seen += 1
        if cid in skip_ids:
            n_skip_existing += 1
            continue
        try:
            mtime = npz.stat().st_mtime
        except FileNotFoundError:
            n_skip_other += 1
            continue
        if now - mtime < min_age_seconds:
            n_skip_other += 1
            continue

        pooled = pool_npz(npz)
        if pooled is None:
            n_skip_other += 1
            continue

        batch.append(
            (
                cid,
                pooled["s_prot_mean"],
                pooled["s_lig_mean"],
                pooled["z_if_mean"],
                pooled["z_if_max"],
                rcycle,
                str(npz),
            )
        )
        n_done += 1

        if len(batch) >= batch_size:
            upsert_batch(conn, batch)
            print(f"[{source}] upserted {n_done} (latest cid={cid})")
            batch.clear()

    upsert_batch(conn, batch)
    print(
        f"[{source}] done: processed={n_done}, "
        f"skipped_existing={n_skip_existing}, skipped_other={n_skip_other}, "
        f"seen={seen}"
    )
    return n_done, n_skip_existing, n_skip_other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["historical", "fast", "fast-r3", "both", "all-r3"],
        default="both",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-pool and upsert even compound_ids already in the table",
    )
    ap.add_argument(
        "--min-age-seconds",
        type=int,
        default=60,
        help=(
            "Skip npz files modified within this many seconds (avoids "
            "racing the in-progress Boltz run)"
        ),
    )
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    conn = psycopg2.connect(**DB_PARAMS)
    try:
        if args.source in ("historical", "both", "all-r3"):
            process_source(
                conn,
                source="historical",
                iterator=iter_historical(conn),
                rcycle=3,
                force=args.force,
                min_age_seconds=args.min_age_seconds,
                batch_size=args.batch_size,
            )
        if args.source in ("fast", "both"):
            process_source(
                conn,
                source="fast",
                iterator=iter_fast(),
                rcycle=1,
                force=args.force,
                min_age_seconds=args.min_age_seconds,
                batch_size=args.batch_size,
            )
        if args.source in ("fast-r3", "all-r3"):
            process_source(
                conn,
                source="fast-r3",
                iterator=iter_fast_r3(),
                rcycle=3,
                force=args.force,
                min_age_seconds=args.min_age_seconds,
                batch_size=args.batch_size,
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT recycling_steps, COUNT(*)
                FROM compound_boltz2_trunk_fast
                GROUP BY recycling_steps
                ORDER BY recycling_steps
                """
            )
            print("\nFinal counts in compound_boltz2_trunk_fast:")
            for rcycle, n in cur.fetchall():
                print(f"  rcycle={rcycle}: {n}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
