"""Generate submission for drop_att_gate (7-pool) variant.

Drops only attentivefp + gatedgcn (keeps pooled_boltz_allpairs) — a
LB A/B vs id=41 (drop_att_gate_pb 6-pool, +0.0057 LB regress) and
id=32 (9-pool baseline). Estimated chemprop family share ~0.78
(marginal zone), filling the unmeasured gap between 0.76 (baseline)
and 0.94 (id=41 catastrophe).

Pipeline:
  1. Monkeypatch ENSEMBLE_MODELS to 7-pool
  2. Run run_ensemble.main() — writes ens_caruana_bag20.csv + DB row
  3. Run run_ensemble_calibrate_importance.main() — writes
     ens_caruana_bag20_calibrated_importance.csv

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/35_drop_att_gate_submit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

import run_ensemble  # noqa: E402
import run_ensemble_calibrate_importance  # noqa: E402

ATTENTIVEFP = "tabpfn_attentivefp_pretrain_embed_umap_default"
GATEDGCN = "tabpfn_gatedgcn_pretrain_embed_umap_default"


def main() -> None:
    base = tuple(run_ensemble.ENSEMBLE_MODELS)
    assert ATTENTIVEFP in base, f"{ATTENTIVEFP} missing"
    assert GATEDGCN in base, f"{GATEDGCN} missing"

    pool_7 = tuple(m for m in base if m not in (ATTENTIVEFP, GATEDGCN))
    print(f"Base pool: {len(base)} -> drop_att_gate: {len(pool_7)} members")
    for m in pool_7:
        print(f"  {m}")

    print("\n===== Step 1: run_ensemble.main() with 7-pool =====")
    orig = run_ensemble.ENSEMBLE_MODELS
    run_ensemble.ENSEMBLE_MODELS = pool_7
    try:
        run_ensemble.main()
    finally:
        run_ensemble.ENSEMBLE_MODELS = orig

    print("\n===== Step 2: importance calibrator =====")
    run_ensemble_calibrate_importance.main()

    out = REPO_ROOT.joinpath(
        "track1_activity",
        "submissions",
        "ens_caruana_bag20_calibrated_importance.csv",
    )
    print(f"\nReady to submit: {out}")


if __name__ == "__main__":
    main()
