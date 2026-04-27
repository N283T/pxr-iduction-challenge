"""Phase 4 caruana SWAP test: Optuna-tuned chemprop pretrain (Trial 10/11).

After Phase 3 confirmed Trial 10/11 ensemble OOF MAE 0.4016/0.4024 beat
production 10-seed (0.4056), test whether SWAP/ADD into the current
9-pool improves caruana_bag20 OOF.

Variants:
  baseline_10seed       current production (deployed in id=32)
  swap_default_t10      seed10ens default -> optuna_trial10 (5-seed)
  swap_default_t11      seed10ens default -> optuna_trial11 (5-seed)
  swap_top500_t10       seed10ens top500  -> optuna_trial10 (5-seed)
  add_t10               ADD optuna_trial10 (10-pool)
  add_t11               ADD optuna_trial11 (10-pool)
  add_both              ADD both (11-pool)
  swap_default_add_t11  default->t10 + ADD t11

Acceptance: caruana_bag20 Δ MAE ≤ -0.003 OR Spearman Δ ≥ +0.002 AND
no member weight > 0.40.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/31_optuna_pretrain_swap.py
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

CHEME_DEFAULT_10 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default"
CHEME_TOP500_10 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
CHEME_OPTUNA_T10 = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default"
)
CHEME_OPTUNA_T11 = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial11_seed5ens_umap_default"
)


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
    except Exception as e:  # capture partial caruana output on later strategy failure
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
    assert CHEME_DEFAULT_10 in base, f"{CHEME_DEFAULT_10} not found"
    assert CHEME_TOP500_10 in base, f"{CHEME_TOP500_10} not found"

    variants = [
        ("baseline_10seed", base),
        ("swap_default_t10", swap(base, CHEME_DEFAULT_10, CHEME_OPTUNA_T10)),
        ("swap_default_t11", swap(base, CHEME_DEFAULT_10, CHEME_OPTUNA_T11)),
        ("swap_top500_t10", swap(base, CHEME_TOP500_10, CHEME_OPTUNA_T10)),
        ("add_t10", add(base, CHEME_OPTUNA_T10)),
        ("add_t11", add(base, CHEME_OPTUNA_T11)),
        ("add_both", add(add(base, CHEME_OPTUNA_T10), CHEME_OPTUNA_T11)),
        (
            "swap_default_add_t11",
            add(swap(base, CHEME_DEFAULT_10, CHEME_OPTUNA_T10), CHEME_OPTUNA_T11),
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
