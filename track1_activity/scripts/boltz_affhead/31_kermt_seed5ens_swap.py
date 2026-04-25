"""kermt multi-seed: caruana SWAP bakeoff (kermt single -> kermt seed5ens).

Plan A extension to a different family (graph transformer / GROVER-base).
4 additional kermt pretrain seeds [43..46] + the production seed (KERMT
default seed=0). Per-row mean of 5 embedding parquets (3200d each).

Single-model OOF deltas (vs baseline kermt_pretrain_embed seed=0):
  MAE:      0.4485 -> 0.4455  (Δ -0.0030)
  Spearman: 0.789  -> 0.798   (Δ +0.0090)

Both move in the right direction; kermt was already pool weight 0.119
(3rd highest after the two cheme seed10ens variants), so even modest
single-member gain should propagate to caruana.

Variants:
  baseline_seed10        current 10-seed deployed pool (id=32, rank 1)
  swap_kermt_seed5ens    swap kermt_pretrain_embed -> kermt_pretrain_embed_seed5ens
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

KERMT_SEED0 = "tabpfn_kermt_pretrain_embed_umap_default"
KERMT_SEED5 = "tabpfn_kermt_pretrain_embed_seed5ens_umap_default"


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
    print(f"Base pool: {len(base)} members (10-seed deployed)")
    assert KERMT_SEED0 in base, f"Expected {KERMT_SEED0} in pool"

    swap_kermt = swap(base, KERMT_SEED0, KERMT_SEED5)

    variants = [
        ("baseline_seed10", base),
        ("swap_kermt_seed5ens", swap_kermt),
    ]
    results = [run_variant(n, p) for n, p in variants]

    print("\n===== Summary (caruana_bag20) =====")
    base_mae = results[0].get("MAE")
    base_sp = results[0].get("Spearman")
    print(f"  {'variant':>22s} (n={'pool':>4s})   MAE       Sp        Δ_MAE     Δ_Sp")
    for r in results:
        mae = r.get("MAE")
        sp = r.get("Spearman")
        if mae is None or sp is None:
            print(f"  {r['name']:>22s}: missing keys, available: {list(r)}")
            continue
        d_mae = mae - base_mae if base_mae is not None else 0.0
        d_sp = sp - base_sp if base_sp is not None else 0.0
        print(
            f"  {r['name']:>22s} (n={r['n_pool']:2d})    "
            f"{mae:.4f}    {sp:.4f}    {d_mae:+.4f}   {d_sp:+.4f}"
        )


if __name__ == "__main__":
    main()
