"""15-seed extension: caruana SWAP bakeoff (10-seed -> 15-seed).

Plan A 15-seed extension (2026-04-25 evening). 5 more chemprop pretrain
seeds [52..56] on top of [42..51]. Variance reduction tapering as expected:

  default:  seed10 MAE 0.4056 Sp 0.840  ->  seed15 MAE 0.4059 Sp 0.840
            (taper hit, +0.0003 MAE, Sp same)
  top500:   seed10 MAE 0.3968 Sp 0.846  ->  seed15 MAE 0.3961 Sp ~0.846
            (slight gain Δ -0.0007)

Variants tested:
  baseline_seed10    current 10-seed candidate (after deploy edit)
  swap_default_only  default seed10 -> seed15
  swap_top500_only   top500 seed10 -> seed15
  swap_both          both swaps simultaneously
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

CHEME_DEFAULT_10 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default"
CHEME_DEFAULT_15 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed15ens_umap_default"
CHEME_TOP500_10 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
CHEME_TOP500_15 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed15ens_top500_umap"


def swap(pool, old, new):
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
    print(f"Base pool: {len(base)} members (10-seed deployed candidate)")
    assert CHEME_DEFAULT_10 in base
    assert CHEME_TOP500_10 in base

    swap_default = swap(base, CHEME_DEFAULT_10, CHEME_DEFAULT_15)
    swap_top500 = swap(base, CHEME_TOP500_10, CHEME_TOP500_15)
    swap_both = swap(swap_default, CHEME_TOP500_10, CHEME_TOP500_15)

    variants = [
        ("baseline_seed10", base),
        ("swap_default_only", swap_default),
        ("swap_top500_only", swap_top500),
        ("swap_both", swap_both),
    ]
    results = [run_variant(n, p) for n, p in variants]

    print("\n===== Summary (caruana_bag20) =====")
    base_mae = results[0].get("MAE")
    base_sp = results[0].get("Spearman")
    print(f"  {'variant':>20s} (n={'pool':>4s})   MAE       Sp        Δ_MAE     Δ_Sp")
    for r in results:
        mae = r.get("MAE")
        sp = r.get("Spearman")
        if mae is None or sp is None:
            print(f"  {r['name']:>20s}: missing keys, available: {list(r)}")
            continue
        d_mae = mae - base_mae if base_mae is not None else 0.0
        d_sp = sp - base_sp if base_sp is not None else 0.0
        print(
            f"  {r['name']:>20s} (n={r['n_pool']:2d})    "
            f"{mae:.4f}    {sp:.4f}    {d_mae:+.4f}   {d_sp:+.4f}"
        )


if __name__ == "__main__":
    main()
