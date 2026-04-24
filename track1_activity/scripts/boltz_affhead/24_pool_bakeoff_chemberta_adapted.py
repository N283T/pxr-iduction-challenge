"""Phase B2 caruana bakeoff: adapted ChemBERTa-5m-MTR vs raw.

Phase B2 (2026-04-25):
- Continued-pretrain ChemBERTa-5m-MTR on log2fc via LoRA -> single-model
  MAE 0.4973 (vs raw 0.5287, Δ -0.031 improvement).
- BUT residual r rose: min 0.77 -> 0.83, max 0.83 -> 0.89. attentivefp
  is now the closest at r=0.89.
- Gate 1 (min-r <= 0.85) still holds (0.83). Gate 2 (single MAE within
  +0.05 of pool weakest = 0.536) easy pass (0.497).

Variants:
  baseline_9pool          current production
  add_raw_5m_mtr          9-pool + raw ChemBERTa (reference)
  add_adapted_5m_mtr      9-pool + continued-pretrained ChemBERTa
  swap_to_adapted         9-pool -attentivefp +adapted (try displacing
                          the most-correlated member to reduce redundancy)
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

RAW = "tabpfn_chemberta_5m_mtr_umap_default"
ADAPTED = "tabpfn_chemberta_5m_mtr_pretrain_embed_umap_default"
AFP = "tabpfn_attentivefp_pretrain_embed_umap_default"


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
        ("add_raw_5m_mtr", BASE + (RAW,)),
        ("add_adapted_5m_mtr", BASE + (ADAPTED,)),
        (
            "swap_afp_to_adapted",
            tuple(m for m in BASE if m != AFP) + (ADAPTED,),
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
