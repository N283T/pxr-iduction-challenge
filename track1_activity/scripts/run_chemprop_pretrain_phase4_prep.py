#!/usr/bin/env -S pixi run python
"""Phase 4 prep: train missing seed=46 + build seed5ens parquets.

Phase 2/3 already trained seeds [42,43,44,45] for top trials. Phase 4
deploy needs seed=46 to complete the [42..46] ensemble (matching the
production seed5ens recipe). Then averages all 5 seeds into a single
log2fc parquet at the canonical location for run_train.py to consume.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_pretrain_phase4_prep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from run_chemprop_pretrain_optuna import (  # noqa: E402
    CKPT_BASE,
    DATA_BASE,
    predict_log2fc,
    pretrain_one,
)
from run_chemprop_pretrain_phase3 import (  # noqa: E402
    build_seed_ensemble_parquet,
    load_top_trials,
)

# Trials selected for Phase 4 deploy (top 2 by ensemble MAE/Sp from Phase 3)
DEPLOY_TRIALS = [10, 11]
DEPLOY_SEEDS = [42, 43, 44, 45, 46]

# Output parquet paths consumed by run_train.py via the new feature
# pattern `cheme_2d_full_boltz_log2fc_pred_optuna_trial{N}_seed5ens`.
DATA_DIR = REPO_ROOT.joinpath("data")


def parquet_path_for_trial(trial_num: int) -> Path:
    return DATA_DIR.joinpath(
        f"chemprop_pretrain_log2fc_predictions_optuna_trial{trial_num}_seed5ens.parquet"
    )


def main() -> None:
    storage_path = DATA_BASE.joinpath("optuna.db")
    storage_url = f"sqlite:///{storage_path}"

    # Load top params from study (need top-N including 11; load 5 for safety)
    top = load_top_trials("log2fc_optuna_v1", storage_url, top_n=15)
    by_num = {row["trial_number"]: row for row in top}
    print(f"Loaded {len(top)} trials from study; deploy targets: {DEPLOY_TRIALS}")

    for trial_num in DEPLOY_TRIALS:
        if trial_num not in by_num:
            raise SystemExit(f"Trial {trial_num} not in top loaded trials")
        params = by_num[trial_num]["params"]
        phase2_mae = by_num[trial_num]["value"]
        print(f"\n===== Trial {trial_num} (Phase 2 OOF MAE {phase2_mae:.4f}) =====")
        print(f"  params: {params}")

        seed_parquets: list[Path] = []
        for seed in DEPLOY_SEEDS:
            t0 = time.time()
            ckpt_dir = (
                CKPT_BASE.joinpath(f"trial_{trial_num:03d}")
                if seed == 42
                else CKPT_BASE.joinpath(f"trial_{trial_num:03d}_seed{seed}")
            )
            log2fc_parquet = (
                DATA_BASE.joinpath(f"trial_{trial_num:03d}_log2fc.parquet")
                if seed == 42
                else DATA_BASE.joinpath(
                    f"trial_{trial_num:03d}_seed{seed}_log2fc.parquet"
                )
            )

            if log2fc_parquet.exists():
                print(f"  seed={seed}: SKIP (parquet exists at {log2fc_parquet.name})")
            else:
                print(f"  seed={seed}: training pretrain + predict_log2fc")
                pretrain_one(params, seed=seed, ckpt_dir=ckpt_dir)
                ckpt_path = ckpt_dir.joinpath("pretrain.pt")
                predict_log2fc(ckpt_path, log2fc_parquet)
                print(f"    seed={seed} done in {time.time() - t0:.0f}s")

            seed_parquets.append(log2fc_parquet)

        # Build seed5ens parquet at canonical location
        out_path = parquet_path_for_trial(trial_num)
        build_seed_ensemble_parquet(seed_parquets, out_path)

        # Sanity check
        import pandas as pd

        df = pd.read_parquet(out_path)
        print(
            f"  seed5ens parquet saved: {out_path.name}  shape={df.shape}  "
            f"mean_8p25={df.log2fc_8p25_pred.mean():.3f}  "
            f"mean_33={df.log2fc_33_pred.mean():.3f}"
        )

    print("\n=== Phase 4 prep done ===")
    for trial_num in DEPLOY_TRIALS:
        print(f"  Trial {trial_num}: {parquet_path_for_trial(trial_num)}")
    print(
        "\nNext: edit run_train.py to add feature pattern "
        "`cheme_2d_full_boltz_log2fc_pred_optuna_trial{N}_seed5ens`, then "
        "run TabPFN UMAP CV for each."
    )


if __name__ == "__main__":
    main()
