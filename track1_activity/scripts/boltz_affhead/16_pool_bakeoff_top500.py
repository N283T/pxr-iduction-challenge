"""Bakeoff: does `tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap`
earn its keep in the caruana pool?

This member strictly supersedes the existing pool anchor
`tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default` on single-model
OOF (MAE 0.4181 vs 0.4212, Δ -0.0031). Question: does swapping the
pool member actually improve caruana_bag20, or is the 0.003 marginal
benefit absorbed by already-weighted siblings?

Variants:
  baseline_10pool          current production ENSEMBLE_MODELS
  add_top500               11-pool (ADD top500 alongside cheme_2df_lf)
  swap_cheme_to_top500     10-pool (SWAP cheme_2df_lf -> top500)
  add_and_swap_dual_boltz  11-pool ADD + drop 2 boltz members that were
                           only -0.0004 improvement (c_concat + raw_plus)
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

TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap"
CHEME = "tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default"
RAW_PLUS = "tabpfn_boltz_raw_plus_pretrain_concat_umap_default"
C_CONCAT = "tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default"


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
    BASE = tuple(run_ensemble.ENSEMBLE_MODELS)
    print(f"Base pool: {len(BASE)} members")
    assert CHEME in BASE, "cheme_2df_lf expected in current ENSEMBLE_MODELS"

    variants = [
        ("baseline_10pool", BASE),
        ("add_top500", BASE + (TOP500,)),
        (
            "swap_cheme_to_top500",
            tuple(m for m in BASE if m != CHEME) + (TOP500,),
        ),
        (
            "add_top500_drop_c_concat",
            tuple(m for m in BASE if m != C_CONCAT) + (TOP500,),
        ),
        (
            "add_top500_drop_raw_plus",
            tuple(m for m in BASE if m != RAW_PLUS) + (TOP500,),
        ),
        (
            "add_top500_drop_both_boltz_strategy3",
            tuple(m for m in BASE if m not in (C_CONCAT, RAW_PLUS)) + (TOP500,),
        ),
    ]

    rows = [run_variant(n, p) for n, p in variants]

    print("\n" + "=" * 100)
    hdr = f"{'variant':<40} {'n':<4} {'MAE':<8} {'ΔMAE':<10} {'RAE':<8} {'Sp':<8} {'w_top500':<10}"
    print(hdr)
    print("-" * 100)
    base_mae = next(
        r.get("MAE", float("nan")) for r in rows if r["name"] == "baseline_10pool"
    )
    for r in rows:
        mae = r.get("MAE", float("nan"))
        rae = r.get("RAE", float("nan"))
        sp = r.get("Spearman", float("nan"))
        w = r.get(f"w_{TOP500}", 0.0)
        delta = mae - base_mae
        marker = " ⭐" if delta < -1e-4 else "  " if delta < 1e-4 else " ✗"
        print(
            f"{r['name']:<40} {r['n_pool']:<4} {mae:.4f}  {delta:+.4f}{marker} "
            f"{rae:.4f}  {sp:.4f}  {w:.4f}"
        )


if __name__ == "__main__":
    main()
