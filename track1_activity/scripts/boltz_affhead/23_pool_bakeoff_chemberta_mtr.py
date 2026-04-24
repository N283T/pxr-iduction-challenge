"""Phase B1.5 caruana bakeoff: ChemBERTa-5m-MTR and/or 10m-MTR add value?

Phase B1 9-variant audit (issue #100 follow-up, 2026-04-24 evening):
- All 9 variants pass min-r <= 0.85 decorrelation gate
- Only chemberta_5m_mtr passes single-OOF within +0.05 of pool weakest
  (MAE 0.5287 vs threshold 0.5360)
- chemberta_10m_mtr near-miss (MAE 0.5367, +0.0007 over threshold)

Per the tier-0 post-mortem, gate3 (caruana ADD Δ <= -0.003) is the
authoritative test. A +0.0007 gate2 miss might still pass caruana.
Test both top candidates.

Variants:
  baseline_9pool       current production
  add_5m_mtr           9-pool + chemberta_5m_mtr
  add_10m_mtr          9-pool + chemberta_10m_mtr
  add_both             10-pool + both MTR variants

Decision gate: Δ caruana MAE <= -0.003 AND caruana weight >= 0.02 on
new member.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

import run_ensemble  # noqa: E402

CB_5M_MTR = "tabpfn_chemberta_5m_mtr_umap_default"
CB_10M_MTR = "tabpfn_chemberta_10m_mtr_umap_default"


def run_variant(name: str, pool: tuple[str, ...]) -> dict:
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
    BASE = tuple(run_ensemble.ENSEMBLE_MODELS)
    print(f"Base pool: {len(BASE)} members")

    variants = [
        ("baseline_9pool", BASE),
        ("add_5m_mtr", BASE + (CB_5M_MTR,)),
        ("add_10m_mtr", BASE + (CB_10M_MTR,)),
        ("add_both", BASE + (CB_5M_MTR, CB_10M_MTR)),
    ]

    results = [run_variant(name, pool) for name, pool in variants]
    print("\n===== Summary =====")
    for r in results:
        parts = " ".join(
            f"{k}={r[k]:.4f}" for k in ("MAE", "RAE", "Spearman_R") if k in r
        )
        print(f"  {r['name']:>24s}  {parts}")


if __name__ == "__main__":
    main()
