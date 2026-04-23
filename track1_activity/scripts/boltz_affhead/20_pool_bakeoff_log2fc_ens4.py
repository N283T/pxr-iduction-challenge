"""Phase 4 of issue #115: caruana bakeoff for ens4 log2fc swap.

Compares 4 pool variants to decide whether the 4-encoder inverse-val-loss
weighted log2fc ensemble (see build_ensemble_log2fc.py) should replace
the chemprop-only log2fc inside cheme_2d_full_boltz_log2fc_pred
(used by 64% of caruana weight).

Variants:
  baseline_9pool                current production ENSEMBLE_MODELS
  swap_default_ens4             replace *_default with *_ens4_umap_default
  swap_top500_ens4              replace *_top500 with *_ens4_top500
  swap_both_ens4                replace both default and top500 with ens4
                                counterparts

Pass acceptance gate (per #115):
  - ens4-inclusive OOF MAE Δ <= -0.005 vs baseline (or <= +0.003 if
    decorrelation value is clear from residual r analysis)
  - caruana weight on cheme family stays < 0.60 (no concentration-blowup)
  - leak-check framework passes on the new log2fc parquet
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

CHEME_DEFAULT = "tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default"
CHEME_ENS4_DEFAULT = "tabpfn_cheme_2d_full_boltz_log2fc_pred_ens4_umap_default"
CHEME_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap"
CHEME_ENS4_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_ens4_top500_umap"


def swap(pool: tuple[str, ...], old: str, new: str) -> tuple[str, ...]:
    assert old in pool, f"{old} not in pool"
    return tuple(new if m == old else m for m in pool)


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
    assert CHEME_DEFAULT in BASE
    assert CHEME_TOP500 in BASE

    variants = [
        ("baseline_9pool", BASE),
        ("swap_default_ens4", swap(BASE, CHEME_DEFAULT, CHEME_ENS4_DEFAULT)),
        ("swap_top500_ens4", swap(BASE, CHEME_TOP500, CHEME_ENS4_TOP500)),
        (
            "swap_both_ens4",
            swap(
                swap(BASE, CHEME_DEFAULT, CHEME_ENS4_DEFAULT),
                CHEME_TOP500,
                CHEME_ENS4_TOP500,
            ),
        ),
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
