"""Gate-1 residual correlation audit for the ChemFM-1B candidates.

Loads OOF predictions from experiment_oof_predictions for the 9 current
pool members plus the two chemfm candidates, then reports Pearson r of
residuals (pred - y_true) between each chemfm variant and every pool
member.

Gate-1 decorrelation threshold (Codex, post-tier-0): min r <= 0.85.
"""

from __future__ import annotations

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
CANDIDATES = (
    "tabpfn_chemfm_1b_last_umap_default",
    "tabpfn_chemfm_1b_mean_umap_default",
)


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
    if not rows:
        raise ValueError(f"No OOF predictions for {name}")
    idx = np.array([r[0] for r in rows])
    pred = np.array([r[1] for r in rows], dtype=np.float64)
    assert (idx == np.arange(len(idx))).all(), f"Non-contiguous idx for {name}"
    return pred


def load_y() -> np.ndarray:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT compound_id, pec50 FROM train_activity ORDER BY id")
    y = np.array([r[1] for r in cur.fetchall()], dtype=np.float64)
    cur.close()
    conn.close()
    return y


def main() -> None:
    base = tuple(run_ensemble.ENSEMBLE_MODELS)
    y = load_y()
    print(f"y_train: n={len(y)}, range=[{y.min():.3f}, {y.max():.3f}]")

    pool_oof = {name: load_oof(name) for name in base}
    cand_oof = {}
    for name in CANDIDATES:
        try:
            cand_oof[name] = load_oof(name)
        except ValueError as e:
            print(f"SKIP {name}: {e}")

    if not cand_oof:
        print("No candidate OOFs available; run TabPFN CV first.")
        return

    pool_res = {n: (p - y) for n, p in pool_oof.items()}
    cand_res = {n: (p - y) for n, p in cand_oof.items()}

    # Single-model MAE
    print("\n===== Single-model OOF MAE =====")
    for name, oof in {**pool_oof, **cand_oof}.items():
        mae = np.mean(np.abs(oof - y))
        print(f"  {name[:55]:55s}  MAE={mae:.4f}")

    pool_mae = {n: float(np.mean(np.abs(p - y))) for n, p in pool_oof.items()}
    weakest = max(pool_mae, key=lambda k: pool_mae[k])
    print(
        f"\nPool weakest: {weakest} (MAE={pool_mae[weakest]:.4f}). "
        f"Gate-2 threshold = {pool_mae[weakest] + 0.05:.4f}"
    )

    # Residual r matrix (candidate vs each pool member)
    print("\n===== Gate 1: residual Pearson r (candidate -> each pool member) =====")
    header = f"{'candidate':>40s}  " + "  ".join(f"{n[:22]:>22s}" for n in base)
    print(header)
    for cname, cres in cand_res.items():
        vals = []
        for pname in base:
            pres = pool_res[pname]
            r = stats.pearsonr(cres, pres).statistic
            vals.append(r)
        vals_arr = np.array(vals)
        row = f"{cname[:40]:>40s}  " + "  ".join(f"{v:>22.3f}" for v in vals)
        print(row)
        print(
            f"  -> min r = {vals_arr.min():.3f}, max r = {vals_arr.max():.3f}  "
            f"[{'PASS' if vals_arr.max() <= 0.85 else 'FAIL'} gate-1]"
        )


if __name__ == "__main__":
    main()
