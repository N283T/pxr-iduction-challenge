"""Bakeoff helper: iterate pool configurations for the Boltz strategy-3
new members (raw_plus_pretrain_concat, c_concat) and report caruana_bag20
OOF MAE per variant.

Does NOT touch ENSEMBLE_MODELS on disk. Instead it monkey-patches the
run_ensemble module in-process so the 8-pool allow-list stays the source
of truth; the iteration is ephemeral.

All variants run sequentially (caruana_bag20 is CPU and ~30-60s per run);
output is a summary table with each variant's MAE/RAE/Spearman and the
new-member weight.
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

RAW_PLUS = "tabpfn_boltz_raw_plus_pretrain_concat_umap_default"
C_CONCAT = "tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default"
POOLED = "tabpfn_pooled_boltz_umap_default"
POOLED_AP = "tabpfn_pooled_boltz_allpairs_umap_default"


def run_variant(name: str, pool: tuple[str, ...]) -> dict:
    """Run caruana_bag20 on a single pool config and return metrics."""
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
    # Parse caruana_bag20 MAE / RAE / Spearman + weights
    metrics = {"name": name, "n_pool": len(pool)}
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "ens_caruana_bag20:" in line:
            for j in range(i + 1, min(i + 20, len(lines))):
                if "OOF RAE=" in lines[j]:
                    parts = lines[j].split()
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            metrics[k.lstrip("OOF ").strip()] = float(v)
                if "Weights" in lines[j]:
                    wstart = j + 1
                    while wstart < len(lines) and "  " in lines[wstart]:
                        w_line = lines[wstart].strip()
                        if " " in w_line and "." in w_line:
                            pieces = w_line.rsplit(None, 1)
                            if len(pieces) == 2:
                                try:
                                    wt = float(pieces[1])
                                    metrics[f"w_{pieces[0].strip()}"] = wt
                                except ValueError:
                                    pass
                        wstart += 1
                    break
            break
    return metrics


def main() -> None:
    # Base allow-list (minus the new member) — freeze at current source state.
    BASE = tuple(m for m in run_ensemble.ENSEMBLE_MODELS)
    print(f"Base pool (read from source): {len(BASE)} members")
    for m in BASE:
        print(f"  {m}")

    variants = [
        ("baseline_8pool", BASE),
        # raw_plus_pretrain_concat variants
        ("add_raw_plus", BASE + (RAW_PLUS,)),
        (
            "swap_pooled_to_raw_plus",
            tuple(m for m in BASE if m != POOLED) + (RAW_PLUS,),
        ),
        (
            "swap_allpairs_to_raw_plus",
            tuple(m for m in BASE if m != POOLED_AP) + (RAW_PLUS,),
        ),
        (
            "swap_dual_to_raw_plus",
            tuple(m for m in BASE if m not in (POOLED, POOLED_AP)) + (RAW_PLUS,),
        ),
        # c_concat variants
        ("add_c_concat", BASE + (C_CONCAT,)),
        (
            "swap_pooled_to_c_concat",
            tuple(m for m in BASE if m != POOLED) + (C_CONCAT,),
        ),
        (
            "swap_allpairs_to_c_concat",
            tuple(m for m in BASE if m != POOLED_AP) + (C_CONCAT,),
        ),
        (
            "swap_dual_to_c_concat",
            tuple(m for m in BASE if m not in (POOLED, POOLED_AP)) + (C_CONCAT,),
        ),
        # both new members
        ("add_both", BASE + (RAW_PLUS, C_CONCAT)),
        (
            "swap_dual_to_both",
            tuple(m for m in BASE if m not in (POOLED, POOLED_AP))
            + (RAW_PLUS, C_CONCAT),
        ),
    ]

    rows = []
    for vname, pool in variants:
        metrics = run_variant(vname, pool)
        rows.append(metrics)
        mae = metrics.get("MAE", float("nan"))
        rae = metrics.get("RAE", float("nan"))
        sp = metrics.get("Spearman", float("nan"))
        w_new = metrics.get(f"w_{RAW_PLUS}", 0.0) + metrics.get(f"w_{C_CONCAT}", 0.0)
        print(
            f"  -> {vname:<30} MAE={mae:.4f}  RAE={rae:.4f}  Sp={sp:.4f}  w_new={w_new:.4f}"
        )

    # Summary table
    print("\n" + "=" * 90)
    print(
        f"{'variant':<30} {'n':<4} {'MAE':<8} {'Δ MAE':<10} {'RAE':<8} {'Sp':<8} {'w_new':<8}"
    )
    print("-" * 90)
    base_mae = next(
        r.get("MAE", float("nan")) for r in rows if r["name"] == "baseline_8pool"
    )
    for r in rows:
        mae = r.get("MAE", float("nan"))
        delta = mae - base_mae
        rae = r.get("RAE", float("nan"))
        sp = r.get("Spearman", float("nan"))
        w_new = r.get(f"w_{RAW_PLUS}", 0.0) + r.get(f"w_{C_CONCAT}", 0.0)
        marker = (
            "  ⭐"
            if mae < base_mae - 1e-4
            else "  "
            if mae < base_mae + 1e-4
            else "  ✗"
        )
        print(
            f"{r['name']:<30} {r['n_pool']:<4} {mae:.4f}  {delta:+.4f}{marker} {rae:.4f}  {sp:.4f}  {w_new:.4f}"
        )


if __name__ == "__main__":
    main()
