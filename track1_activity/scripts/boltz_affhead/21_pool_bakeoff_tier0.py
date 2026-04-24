"""Phase A caruana bakeoff: can Boltz-2 tier-0 (17d scalars) earn its
keep despite weak single-model OOF?

Single-model OOF MAE 0.5797 -- roughly +0.09 worse than pool's weakest
member (pooled_boltz_allpairs 0.4860). But residual r to all 9 pool
members is 0.71-0.82, well below the 0.85 target Codex flagged for
"new axis" candidates. Current pool is largely GNN-family embedding-
heavy; tier-0 adds post-trunk head scalars as a different family.

Variants:
  baseline_9pool     current production ENSEMBLE_MODELS
  add_tier0          10-pool (ADD tier0 to current 9)

Pass gate (#115 Phase 4 style):
  caruana Δ MAE ≤ -0.002 vs baseline, and pool diversity (cheme family
  weight) stays under 0.60.
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
