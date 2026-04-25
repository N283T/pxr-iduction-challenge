"""Phase A caruana bakeoff: does ChemFM-1B add value to the 9-pool?

ChemFM-1B (TheLuoFengLab, Nature Comm Chem 2025) is a Llama-style
causal LM (22 layers, hidden=2048, vocab=320) pretrained on UniChem
SMILES. Fills a gap in the current pool which has no causal-LM member
(existing BERT-family members are all encoder-only).

Variants:
  baseline_9pool        current production ENSEMBLE_MODELS
  add_chemfm_1b_last    9-pool + chemfm_1b_last
  add_chemfm_1b_mean    9-pool + chemfm_1b_mean
  add_both              9-pool + both poolings

3-gate acceptance (post-tier-0 rules, 2026-04-24):
  gate1 min residual r <= 0.85 vs all 9 pool members
  gate2 single-model OOF MAE within +0.05 of pool weakest (~0.486)
        AND close to pool top tier (~0.42) for best-case LB gain
  gate3 caruana_bag20 Δ MAE <= -0.003 (noise-level is +/- 0.002)
        AND caruana weight on new member >= 0.02
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

CHEMFM_LAST = "tabpfn_chemfm_1b_last_umap_default"
CHEMFM_MEAN = "tabpfn_chemfm_1b_mean_umap_default"


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
    for m in BASE:
        print(f"  {m}")

    variants = [
        ("baseline_9pool", BASE),
        ("add_chemfm_1b_last", BASE + (CHEMFM_LAST,)),
        ("add_chemfm_1b_mean", BASE + (CHEMFM_MEAN,)),
        ("add_both", BASE + (CHEMFM_LAST, CHEMFM_MEAN)),
    ]

    results = [run_variant(name, pool) for name, pool in variants]
    print("\n===== Summary =====")
    for r in results:
        parts = " ".join(
            f"{k}={r[k]:.4f}" for k in ("MAE", "RAE", "Spearman_R") if k in r
        )
        print(f"  {r['name']:>22s} (n={r['n_pool']:2d})  {parts}")

    base_mae = results[0].get("MAE")
    if base_mae is None:
        return
    print(
        "\n===== Δ vs baseline (MAE) =====\n"
        f"  {'variant':>22s}  Δ_MAE     caruana gate (<=-0.003)"
    )
    for r in results[1:]:
        dm = r.get("MAE")
        if dm is None:
            continue
        delta = dm - base_mae
        gate = "PASS" if delta <= -0.003 else "FAIL"
        print(f"  {r['name']:>22s}  {delta:+.4f}   [{gate}]")


if __name__ == "__main__":
    main()
