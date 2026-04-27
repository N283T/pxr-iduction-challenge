"""Caruana drop test: remove low-weight members (att 0.0024 / gate 0.018 /
pooled_boltz_allpairs 0.035) to test pool simplification.

Variants:
  baseline         9-pool current
  drop_att         drop attentivefp (8-pool)
  drop_att_gate    drop attentivefp + gatedgcn (7-pool)
  drop_att_gate_pb drop attentivefp + gatedgcn + pooled_boltz_allpairs (6-pool)

Acceptance (per feedback_caruana_bag_variance_0003): require Δ MAE
<= -0.005 within-run for real signal beyond bag noise +-0.003.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/34_drop_lowweight.py
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
POOLED_BOLTZ_ALLPAIRS = "tabpfn_pooled_boltz_allpairs_umap_default"


def drop(pool: tuple[str, ...], targets: tuple[str, ...]) -> tuple[str, ...]:
    for t in targets:
        assert t in pool, f"{t} not in pool"
    return tuple(m for m in pool if m not in targets)


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
    print(f"Base pool: {len(base)} members (9-pool current)")
    for m in (ATTENTIVEFP, GATEDGCN, POOLED_BOLTZ_ALLPAIRS):
        assert m in base, f"{m} not in base"

    variants = [
        ("baseline", base),
        ("drop_att", drop(base, (ATTENTIVEFP,))),
        ("drop_att_gate", drop(base, (ATTENTIVEFP, GATEDGCN))),
        (
            "drop_att_gate_pb",
            drop(base, (ATTENTIVEFP, GATEDGCN, POOLED_BOLTZ_ALLPAIRS)),
        ),
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
