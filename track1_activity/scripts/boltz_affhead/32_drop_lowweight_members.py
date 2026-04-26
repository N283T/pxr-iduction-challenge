"""Quick drop test: remove low-weight members (gatedgcn, attentivefp).

Post-Phase 4 (10-pool) caruana weights:
  attentivefp ~0.000  (effectively dropped already)
  gatedgcn    ~0.003
  molformer_c3 ~0.012
  pooled_boltz_allpairs ~0.019
  pooled_boltz ~0.020

Hypothesis: pure noise contributors. Caruana already deweights them,
but the bagged search may still leak weight that hurts on test.
Try dropping the bottom 2 (attentivefp, gatedgcn) and see if OOF
moves.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/32_drop_lowweight_members.py
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

ATTENTIVEFP = "tabpfn_attentivefp_pretrain_embed_umap_default"
GATEDGCN = "tabpfn_gatedgcn_pretrain_embed_umap_default"
MOLFORMER_C3 = "tabpfn_molformer_c3_pretrain_embed_umap"


def drop(pool: tuple[str, ...], members: list[str]) -> tuple[str, ...]:
    return tuple(m for m in pool if m not in members)


def run_variant(name: str, pool: tuple[str, ...]) -> dict:
    print(f"\n===== {name} ({len(pool)} members) =====")
    orig = run_ensemble.ENSEMBLE_MODELS
    run_ensemble.ENSEMBLE_MODELS = pool
    buf = io.StringIO()
    err: Exception | None = None
    try:
        with contextlib.redirect_stdout(buf):
            run_ensemble.main()
    except Exception as e:
        err = e
    finally:
        run_ensemble.ENSEMBLE_MODELS = orig
    output = buf.getvalue()
    print(output)
    if err is not None:
        print(f"  [variant {name}] partial: {type(err).__name__}: {err}")
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
    print(f"Base pool: {len(base)} members (Phase 4 deploy, 10-pool)")
    assert ATTENTIVEFP in base, f"{ATTENTIVEFP} not found"
    assert GATEDGCN in base, f"{GATEDGCN} not found"

    variants = [
        ("baseline_10pool", base),
        ("drop_attentivefp", drop(base, [ATTENTIVEFP])),
        ("drop_gatedgcn", drop(base, [GATEDGCN])),
        ("drop_both", drop(base, [ATTENTIVEFP, GATEDGCN])),
        ("drop_three", drop(base, [ATTENTIVEFP, GATEDGCN, MOLFORMER_C3])),
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
