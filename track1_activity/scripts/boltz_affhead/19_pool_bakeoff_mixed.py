"""Bakeoff with mixed_pool_top500 — ADD vs SWAP variants.

Motivated by the 2026-04-23 01:08 LB regression: ADD-only pool update
(10-pool with both cheme_top500 and mixed_top500) produced OOF MAE 0.4108
but LB MAE 0.4200 (+0.0051 worse than previous 0.4149). Caruana
concentrated 68% weight on cheme-derivative members, destroying pool
diversity on LB.

This bakeoff tests multiple SWAP configurations to find a more robust
pool that doesn't over-weight the cheme-feature family.

Note: mixed_pool_top500 source shares:
  KERMT 42.2%, cheme_2df 14.9%, pooled_boltz 14.0%, molformer 14.4%,
  gatedgcn 6.6%, attentivefp 4.7%, chemprop 3.2%
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

MIXED = "tabpfn_mixed_pool_top500_umap"
CHEME_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap"
CHEME_FULL = "tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default"
KERMT = "tabpfn_kermt_pretrain_embed_umap_default"
MOLFORMER = "tabpfn_molformer_c3_pretrain_embed_umap"
POOLED_AP = "tabpfn_pooled_boltz_allpairs_umap_default"
POOLED = "tabpfn_pooled_boltz_umap_default"


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
                if "Weights" in lines[j]:
                    ws = j + 1
                    while ws < len(lines) and "  " in lines[ws]:
                        wl = lines[ws].strip()
                        if " " in wl:
                            pieces = wl.rsplit(None, 1)
                            if len(pieces) == 2:
                                try:
                                    metrics[f"w_{pieces[0].strip()}"] = float(pieces[1])
                                except ValueError:
                                    pass
                        ws += 1
                    break
            break
    return metrics


def main() -> None:
    # Current ENSEMBLE_MODELS in source = 10-pool (already includes MIXED)
    # We restore BASE to the pre-mixed 9-pool for cleaner diff
    CURRENT = tuple(run_ensemble.ENSEMBLE_MODELS)
    PRE_MIXED = tuple(m for m in CURRENT if m != MIXED)
    assert len(PRE_MIXED) == 9, f"expected 9-pool base, got {len(PRE_MIXED)}"
    print(f"Base 9-pool (source minus mixed): {len(PRE_MIXED)} members")

    variants = [
        ("baseline_9pool_no_mixed", PRE_MIXED),
        ("add_mixed (today's submit)", PRE_MIXED + (MIXED,)),
        (
            "swap_cheme_top500_to_mixed",
            tuple(m for m in PRE_MIXED if m != CHEME_TOP500) + (MIXED,),
        ),
        (
            "swap_cheme_full_to_mixed",
            tuple(m for m in PRE_MIXED if m != CHEME_FULL) + (MIXED,),
        ),
        (
            "swap_kermt_to_mixed (KERMT is 42% of mixed)",
            tuple(m for m in PRE_MIXED if m != KERMT) + (MIXED,),
        ),
        (
            "swap_molformer_to_mixed",
            tuple(m for m in PRE_MIXED if m != MOLFORMER) + (MIXED,),
        ),
        (
            "swap_dual_pooled_to_mixed",
            tuple(m for m in PRE_MIXED if m not in (POOLED, POOLED_AP)) + (MIXED,),
        ),
        (
            "swap_cheme_both_to_mixed",
            tuple(m for m in PRE_MIXED if m not in (CHEME_FULL, CHEME_TOP500))
            + (MIXED,),
        ),
    ]

    rows = [run_variant(n, p) for n, p in variants]

    print("\n" + "=" * 100)
    hdr = f"{'variant':<45} {'n':<4} {'MAE':<8} {'RAE':<8} {'Sp':<8} {'w_mixed':<10}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        mae = r.get("MAE", float("nan"))
        rae = r.get("RAE", float("nan"))
        sp = r.get("Spearman", float("nan"))
        w = r.get(f"w_{MIXED}", 0.0)
        print(
            f"{r['name']:<45} {r['n_pool']:<4} {mae:.4f}  {rae:.4f}  {sp:.4f}  {w:.4f}"
        )


if __name__ == "__main__":
    main()
