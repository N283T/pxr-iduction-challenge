#!/usr/bin/env -S pixi run python
"""Phase 3: multi-seed validate top-N Optuna trials.

Loads best N trials from the Phase 2 Optuna study, re-runs each trial's
hparams across multiple seeds, builds per-trial seed-ensemble log2fc
parquet, evaluates downstream TabPFN OOF MAE.

Per memory:feedback_single_seed_dim_sweep_spurious — sub-0.01 OOF MAE
peaks can flip sign across seeds. This phase quantifies inter-seed
variance and ensemble stability before committing to deploy (Phase 4).

Usage:
    # Default: top 3 trials × seeds [43,44,45]
    pixi run python track1_activity/scripts/run_chemprop_pretrain_phase3.py

    # Custom trials and seeds
    pixi run python track1_activity/scripts/run_chemprop_pretrain_phase3.py \
        --top-n 3 --seeds 43,44,45 --study-name log2fc_optuna_v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from run_chemprop_pretrain_optuna import (  # noqa: E402
    CKPT_BASE,
    DATA_BASE,
    evaluate_oof_mae,
    predict_log2fc,
    pretrain_one,
)


def load_top_trials(study_name: str, storage: str, top_n: int) -> list[dict]:
    """Return list of {trial_number, value, params} for top-N completed trials."""
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)  # minimize → ascending
    top = completed[:top_n]
    rows = []
    for t in top:
        # Reconstruct full params dict (Optuna stores categorical/float values)
        params = dict(t.params)
        # Add fixed values from sample_params() that aren't suggested
        params["activation"] = "relu"
        params["warmup_epochs"] = 3
        params["max_epochs"] = 100
        params["patience"] = 10
        rows.append(
            {
                "trial_number": t.number,
                "value": t.value,
                "spearman": t.user_attrs.get("oof_spearman", float("nan")),
                "params": params,
            }
        )
    return rows


def build_seed_ensemble_parquet(seed_parquets: list[Path], out_path: Path) -> None:
    """Average log2fc predictions across seeds, save to out_path."""
    dfs = [pd.read_parquet(p) for p in seed_parquets]
    # Ensure same compound_id order across seeds
    base_idx = dfs[0].index
    for d in dfs[1:]:
        assert (d.index == base_idx).all(), "compound_id mismatch across seeds"
    mean_df = pd.DataFrame(
        {
            "log2fc_8p25_pred": np.mean(
                [d["log2fc_8p25_pred"].values for d in dfs], axis=0
            ),
            "log2fc_33_pred": np.mean(
                [d["log2fc_33_pred"].values for d in dfs], axis=0
            ),
        },
        index=base_idx,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_parquet(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 multi-seed validation")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--seeds", type=str, default="43,44,45", help="comma-separated seeds"
    )
    parser.add_argument("--study-name", type=str, default="log2fc_optuna_v1")
    parser.add_argument("--storage", type=str, default=None)
    args = parser.parse_args()

    if args.storage is None:
        storage_path = DATA_BASE.joinpath("optuna.db")
        args.storage = f"sqlite:///{storage_path}"

    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Loading top {args.top_n} trials from {args.study_name}")
    top = load_top_trials(args.study_name, args.storage, args.top_n)
    print(f"Top {len(top)} trials:")
    for row in top:
        print(
            f"  Trial {row['trial_number']}: OOF MAE={row['value']:.4f}  "
            f"Sp={row['spearman']:.4f}"
        )
    print(f"\nSeeds: {seeds}")

    print("\nLoading train + test")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  train n={len(train_df)}  test n={len(test_df)}")

    results = []
    for row in top:
        trial_num = row["trial_number"]
        params = row["params"]
        print(f"\n===== Trial {trial_num} (Phase 2 OOF MAE {row['value']:.4f}) =====")

        per_seed_oof = []
        seed_parquets = []
        for seed in seeds:
            print(f"\n  --- Trial {trial_num} seed={seed} ---")
            t0 = time.time()
            ckpt_dir = CKPT_BASE.joinpath(f"trial_{trial_num:03d}_seed{seed}")
            log2fc_parquet = DATA_BASE.joinpath(
                f"trial_{trial_num:03d}_seed{seed}_log2fc.parquet"
            )
            if not log2fc_parquet.exists():
                pretrain_one(params, seed=seed, ckpt_dir=ckpt_dir)
                ckpt_path = ckpt_dir.joinpath("pretrain.pt")
                predict_log2fc(ckpt_path, log2fc_parquet)
            else:
                print(f"    SKIP pretrain (exists): {log2fc_parquet}")
            print(f"    seed={seed} pretrain+predict in {time.time() - t0:.0f}s")

            t1 = time.time()
            mae, sp = evaluate_oof_mae(log2fc_parquet, train_df, test_df, y_train)
            print(
                f"    seed={seed} eval done in {time.time() - t1:.0f}s. "
                f"OOF MAE={mae:.4f}  Sp={sp:.4f}"
            )
            per_seed_oof.append((seed, mae, sp))
            seed_parquets.append(log2fc_parquet)

        # Build ensemble parquet across seeds
        ens_parquet = DATA_BASE.joinpath(
            f"trial_{trial_num:03d}_seed{len(seeds)}ens_log2fc.parquet"
        )
        build_seed_ensemble_parquet(seed_parquets, ens_parquet)

        # Evaluate ensemble
        ens_mae, ens_sp = evaluate_oof_mae(ens_parquet, train_df, test_df, y_train)

        # Stats
        per_seed_mae_arr = np.array([r[1] for r in per_seed_oof])
        per_seed_sp_arr = np.array([r[2] for r in per_seed_oof])
        print(f"\n  Trial {trial_num} summary:")
        for seed, mae, sp in per_seed_oof:
            print(f"    seed={seed}: MAE={mae:.4f}  Sp={sp:.4f}")
        print(
            f"  per-seed MAE mean ± std: "
            f"{per_seed_mae_arr.mean():.4f} ± {per_seed_mae_arr.std():.4f}"
        )
        print(
            f"  per-seed Sp  mean ± std: "
            f"{per_seed_sp_arr.mean():.4f} ± {per_seed_sp_arr.std():.4f}"
        )
        print(f"  ENSEMBLE ({len(seeds)} seeds): MAE={ens_mae:.4f}  Sp={ens_sp:.4f}")
        print(
            f"  ensemble vs single-seed phase2: "
            f"Δ MAE = {ens_mae - row['value']:+.4f}  "
            f"Δ Sp = {ens_sp - row['spearman']:+.4f}"
        )

        results.append(
            {
                "trial_number": trial_num,
                "phase2_mae": row["value"],
                "phase2_sp": row["spearman"],
                "per_seed_mae": [r[1] for r in per_seed_oof],
                "per_seed_sp": [r[2] for r in per_seed_oof],
                "per_seed_mae_mean": float(per_seed_mae_arr.mean()),
                "per_seed_mae_std": float(per_seed_mae_arr.std()),
                "ens_mae": ens_mae,
                "ens_sp": ens_sp,
            }
        )

    print("\n\n===== Phase 3 final =====")
    print(
        f"{'Trial':>6} {'Phase2 MAE':>10} {'PerSeed mean±std':>18} "
        f"{'Ensemble MAE':>14} {'Ens Sp':>8}"
    )
    for r in results:
        print(
            f"{r['trial_number']:>6} {r['phase2_mae']:>10.4f} "
            f"{r['per_seed_mae_mean']:>10.4f}±{r['per_seed_mae_std']:.4f}  "
            f"{r['ens_mae']:>14.4f} {r['ens_sp']:>8.4f}"
        )

    # Pick best by ensemble MAE
    best = min(results, key=lambda r: r["ens_mae"])
    print(
        f"\nBest ensemble: Trial {best['trial_number']} "
        f"OOF MAE {best['ens_mae']:.4f}  Sp {best['ens_sp']:.4f}"
    )
    print("\nReference points:")
    print("  Optuna baseline (DEFAULT_PARAMS, max_epochs=100): OOF MAE 0.4246")
    print("  Production single-seed=42 OOF MAE: 0.4213")
    print("  Production 5-seed ensemble OOF MAE: 0.4068")
    print("  Production 10-seed ensemble OOF MAE: 0.4056")
    print(
        "\nIf best ensemble MAE close to or below 0.406, proceed to Phase 4 "
        "(5-seed deploy + caruana SWAP test)."
    )


if __name__ == "__main__":
    main()
