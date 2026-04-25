"""10-seed extension: caruana SWAP bakeoff (5-seed -> 10-seed).

Plan A 10-seed extension (2026-04-25 evening). 5 additional chemprop
pretrain seeds [47..51] on top of the production [42..46]. Both
single-model OOF and Spearman improve incrementally:

  default:  seed5  MAE 0.4068 Sp 0.836  ->  seed10 MAE 0.4056 Sp 0.840
  top500:   seed5  MAE 0.3988 Sp 0.843  ->  seed10 MAE 0.3968 Sp 0.846

Spearman +0.003-0.004 — exactly the gap to rank 2 (sia 0.850 vs our
LB 0.846). MAE -0.001 to -0.002 — within fold-std noise but
directionally good. Both wanted.

Variants:
  baseline_seed5     current production (deployed in id=31, rank 1)
  swap_default_only  seed5 default -> seed10 default
  swap_top500_only   seed5 top500  -> seed10 top500
  swap_both          both swaps simultaneously

Acceptance: caruana_bag20 Δ MAE <= -0.002 OR Spearman Δ >= +0.002 AND
no member weight > 0.40 (concentration risk).
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

CHEME_DEFAULT_5 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_umap_default"
CHEME_DEFAULT_10 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default"
CHEME_TOP500_5 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_top500_umap"
CHEME_TOP500_10 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"


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
    print(f"Base pool: {len(base)} members (5-seed deployed in id=31)")
    assert CHEME_DEFAULT_5 in base
    assert CHEME_TOP500_5 in base

    swap_default = swap(base, CHEME_DEFAULT_5, CHEME_DEFAULT_10)
    swap_top500 = swap(base, CHEME_TOP500_5, CHEME_TOP500_10)
    swap_both = swap(swap_default, CHEME_TOP500_5, CHEME_TOP500_10)

    variants = [
        ("baseline_seed5", base),
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
