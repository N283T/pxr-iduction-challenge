"""Multi-seed log2fc ensemble: residual r audit + caruana SWAP bakeoff.

Phase A1 (2026-04-25 multi-seed pretrain): the seed5ens variant of
cheme_2d_full_boltz_log2fc_pred achieves OOF MAE 0.4069 vs baseline
0.4181 (Δ -0.011, >0.4σ of fold std 0.025). Same encoder family +
hyperparams as the pool's strongest member, only the source-randomness
of pretrain weights differs. Residual correlation with the original
single-seed variant is expected to be very high (>= 0.95), so this is a
SWAP candidate rather than an ADD.

Tests run:
  Gate 1 — residual Pearson r between seed5ens and each 9-pool member.
           SWAP target's r should be near 1.0 (highly redundant);
           others give a sense of whether seed5ens decorrelates from
           the rest of the pool.
  Gate 3 — caruana_bag20 OOF MAE for two pool configurations:
             baseline   = current 9-pool (with cheme_2d_full_boltz_log2fc_pred)
             swap_seed5 = swap cheme_2d_full_boltz_log2fc_pred for the
                          seed5ens variant
           Decision rule: SWAP wins if Δ caruana MAE <= -0.003 AND no
           pool member jumps to weight > 0.40 (concentration risk).

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/27_seed5ens_swap_audit.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

import run_ensemble  # noqa: E402

DB = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}
CHEME_BASELINE = "tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default"
CHEME_SEED5 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_umap_default"


def load_oof(name: str) -> np.ndarray:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT id FROM experiments WHERE name = %s", (name,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Experiment not found: {name}")
    exp_id = row[0]
    cur.execute(
        "SELECT train_idx, oof_prediction FROM experiment_oof_predictions "
        "WHERE experiment_id = %s ORDER BY train_idx",
        (exp_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return np.array([r[1] for r in rows], dtype=np.float64)


def load_y() -> np.ndarray:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT compound_id, pec50 FROM train_activity ORDER BY id")
    y = np.array([r[1] for r in cur.fetchall()], dtype=np.float64)
    cur.close()
    conn.close()
    return y


def run_pool_variant(name: str, pool: tuple[str, ...]) -> dict:
    print(f"\n===== {name} ({len(pool)} members) =====")
    orig = run_ensemble.ENSEMBLE_MODELS
    run_ensemble.ENSEMBLE_MODELS = pool
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_ensemble.main()
    finally:
        run_ensemble.ENSEMBLE_MODELS = orig
    output = buf.getvalue()
    print(output)
    metrics = {"name": name, "n_pool": len(pool)}
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "ens_caruana_bag20:" in line:
            for j in range(i + 1, min(i + 30, len(lines))):
                if "OOF RAE=" in lines[j]:
                    for p in lines[j].split():
                        if "=" in p:
                            k, v = p.split("=", 1)
                            try:
                                metrics[k.lstrip("OOF ").strip()] = float(v)
                            except ValueError:
                                pass
            break
    return metrics


def main() -> None:
    base = tuple(run_ensemble.ENSEMBLE_MODELS)
    print(f"Base pool: {len(base)} members")
    assert CHEME_BASELINE in base, (
        f"Expected {CHEME_BASELINE} in pool; current pool: {base}"
    )

    # ---- Gate 1: residual r ----
    y = load_y()
    print(f"\ny_train: n={len(y)}")

    oofs = {n: load_oof(n) for n in base}
    seed5_oof = load_oof(CHEME_SEED5)
    print("Loaded all OOFs.")

    seed5_resid = seed5_oof - y
    print("\n===== Gate 1: residual r — seed5ens vs 9-pool members =====")
    for n in base:
        r = stats.pearsonr(seed5_resid, oofs[n] - y).statistic
        flag = "  <-- SWAP target" if n == CHEME_BASELINE else ""
        print(f"  {n[:55]:55s}  r={r:+.4f}{flag}")

    seed5_mae = float(np.mean(np.abs(seed5_oof - y)))
    print(f"\nseed5ens single OOF MAE: {seed5_mae:.4f}")
    print(
        f"baseline single OOF MAE: "
        f"{float(np.mean(np.abs(oofs[CHEME_BASELINE] - y))):.4f}"
    )

    # ---- Gate 3: caruana SWAP bakeoff ----
    print("\n===== Gate 3: caruana SWAP bakeoff =====")
    swap_pool = tuple(CHEME_SEED5 if m == CHEME_BASELINE else m for m in base)
    variants = [
        ("baseline_9pool", base),
        ("swap_cheme_seed5ens", swap_pool),
    ]
    results = [run_pool_variant(n, p) for n, p in variants]

    print("\n===== Summary =====")
    for r in results:
        parts = " ".join(
            f"{k}={r[k]:.4f}" for k in ("MAE", "RAE", "Spearman_R") if k in r
        )
        print(f"  {r['name']:>22s} (n={r['n_pool']:2d})  {parts}")

    base_mae = results[0].get("MAE")
    if base_mae is None:
        return
    swap_mae = results[1].get("MAE")
    if swap_mae is None:
        return
    delta = swap_mae - base_mae
    gate_pass = delta <= -0.003
    print(
        f"\nSWAP Δ MAE: {delta:+.4f}  "
        f"[{'PASS' if gate_pass else 'fail'} threshold <=-0.003]"
    )


if __name__ == "__main__":
    main()
