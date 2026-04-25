"""Multi-seed log2fc ensemble: double-SWAP caruana bakeoff.

Phase A1 single SWAP (script 27): replacing cheme_2d_full_boltz_log2fc_pred
with its seed5ens variant gives caruana_bag20 OOF MAE 0.4150 -> 0.4097
(Δ -0.0053). Both cheme_top500 and the cheme default were originally
generated from the same single-seed log2fc_pred parquet, so the same
multi-seed averaging trick can be applied to top500 too.

This script tests both single-swap variants and the double-swap:
  baseline_9pool         current production
  swap_default_only      cheme_default -> cheme_default_seed5ens
  swap_top500_only       cheme_top500 -> cheme_top500_seed5ens
  swap_both              both swaps simultaneously

Acceptance: caruana_bag20 Δ MAE <= -0.005 AND no member weight > 0.40
(concentration risk).

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/28_seed5ens_double_swap.py
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
CHEME_DEFAULT_SEED5 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_umap_default"
CHEME_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap"
CHEME_TOP500_SEED5 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_top500_umap"


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
    base = tuple(run_ensemble.ENSEMBLE_MODELS)
    print(f"Base pool: {len(base)} members")
    assert CHEME_DEFAULT in base
    assert CHEME_TOP500 in base

    swap_default = swap(base, CHEME_DEFAULT, CHEME_DEFAULT_SEED5)
    swap_top500 = swap(base, CHEME_TOP500, CHEME_TOP500_SEED5)
    swap_both = swap(swap_default, CHEME_TOP500, CHEME_TOP500_SEED5)

    variants = [
        ("baseline_9pool", base),
        ("swap_default_only", swap_default),
        ("swap_top500_only", swap_top500),
        ("swap_both", swap_both),
    ]
    results = [run_variant(n, p) for n, p in variants]

    print("\n===== Summary (caruana_bag20) =====")
    base_mae = results[0].get("MAE")
    print(f"  {'variant':>22s} (n={'pool':>4s})  MAE      RAE      Δ_vs_base")
    for r in results:
        mae = r.get("MAE")
        rae = r.get("RAE")
        if mae is None or rae is None:
            continue
        delta = mae - base_mae if base_mae is not None else 0.0
        gate = "PASS" if delta <= -0.005 else "fail"
        print(
            f"  {r['name']:>22s} (n={r['n_pool']:2d})    "
            f"{mae:.4f}  {rae:.4f}  {delta:+.4f} [{gate}]"
        )


if __name__ == "__main__":
    main()
