"""Phase B caruana bakeoff: does mordred3d (213d) add on top of tier-0?

tier0 (17d scalars) already joined the pool in Phase A bakeoff (10-pool
MAE 0.4130, baseline 0.4150). Now test whether Boltz-2 Mordred 3D
descriptors (pose-conditioned 213d) add further value.

Residual r (mordred3d vs existing 10-pool): 0.69-0.80, with top500 at
r=0.69 (lowest), tier0 at r=0.80 (highest, expected same family). All
below the 0.85 Codex target.

Variants:
  baseline_9pool     current production
  add_tier0          Phase A (10-pool)
  add_mordred3d      9-pool + mordred3d (skip tier0)
  add_both           11-pool (tier0 + mordred3d)
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

TIER0 = "tabpfn_boltz2_tabular_tier0_umap_default"
MORDRED3D = "tabpfn_boltz2_mordred3d_umap_default"


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
        ("add_tier0", BASE + (TIER0,)),
        ("add_mordred3d", BASE + (MORDRED3D,)),
        ("add_both", BASE + (TIER0, MORDRED3D)),
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
