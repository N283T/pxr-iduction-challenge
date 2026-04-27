"""Caruana SWAP test: chemprop_pretrain_embed -> optuna_trial10/11_embed.

Mirrors 31_optuna_pretrain_swap.py for the embed channel. Tests whether
swapping the default-seed 256d frozen embedding for the Optuna-tuned
384d embedding (Trial 10/11) improves caruana_bag20 OOF.

Variants:
  baseline           current 9-pool (deployed in id=32)
  swap_t10           chemprop_pretrain_embed -> optuna_trial10_embed
  swap_t11           chemprop_pretrain_embed -> optuna_trial11_embed
  add_t10            ADD optuna_trial10_embed (10-pool)
  add_t11            ADD optuna_trial11_embed (10-pool)

Acceptance (per feedback_oof_minus_0002_ceiling +
feedback_caruana_bag_variance_0003): caruana_bag20 Δ MAE <= -0.003
to claim real signal above bag noise.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/33_chemprop_embed_swap.py
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

CHEMPROP_EMBED = "tabpfn_chemprop_pretrain_embed_umap_default"
CHEMPROP_OPTUNA_T10 = "tabpfn_chemprop_pretrain_optuna_trial10_embed_umap"
CHEMPROP_OPTUNA_T11 = "tabpfn_chemprop_pretrain_optuna_trial11_embed_umap"


def swap(pool: tuple[str, ...], old: str, new: str) -> tuple[str, ...]:
    assert old in pool, f"{old} not in pool"
    return tuple(new if m == old else m for m in pool)


def add(pool: tuple[str, ...], new: str) -> tuple[str, ...]:
    assert new not in pool, f"{new} already in pool"
    return tuple(list(pool) + [new])


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
    print(f"Base pool: {len(base)} members (10-seed deployed in id=32)")
    assert CHEMPROP_EMBED in base, f"{CHEMPROP_EMBED} not found"

    variants = [
        ("baseline", base),
        ("swap_t10", swap(base, CHEMPROP_EMBED, CHEMPROP_OPTUNA_T10)),
        ("swap_t11", swap(base, CHEMPROP_EMBED, CHEMPROP_OPTUNA_T11)),
        ("add_t10", add(base, CHEMPROP_OPTUNA_T10)),
        ("add_t11", add(base, CHEMPROP_OPTUNA_T11)),
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
