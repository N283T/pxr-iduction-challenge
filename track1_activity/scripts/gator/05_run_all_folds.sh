#!/usr/bin/env bash
# Run 5-fold head-only FT + val inference for GatorAffinity on PXR pEC50.
#
# Usage:
#   bash 05_run_all_folds.sh <run_name>
#
# Produces structures/gator/ft_runs/<run_name>/fold{0..4}/.../ and
# <run_name>/fold{k}_val_preds.csv, which 06_collate_oof.py then stitches
# into one OOF table for experiment registration.
set -euo pipefail

RUN_NAME="${1:-fold_run_head_v1}"
FORK_DIR=/home/nagaet/ghq/github.com/N283T/GatorAffinity
REPO_DIR=/home/nagaet/pxr-iduction-challenge
OUT_ROOT="${REPO_DIR}/structures/gator/ft_runs/${RUN_NAME}"
mkdir -p "${OUT_ROOT}"

LR_INIT="${LR_INIT:-1e-6}"
LR_WARM="${LR_WARM:-1e-4}"
LR_COS="${LR_COS:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_EPOCH="${MAX_EPOCH:-15}"
WARMUP="${WARMUP:-2}"
PATIENCE="${PATIENCE:-5}"

for K in 0 1 2 3 4; do
  SAVE_DIR="${OUT_ROOT}/fold${K}"
  LOG="${OUT_ROOT}/fold${K}.log"
  if [[ -d "${SAVE_DIR}/version_0/checkpoint" ]]; then
    echo "[fold${K}] already has checkpoint dir, skipping train"
  else
    echo "[fold${K}] training -> ${SAVE_DIR}"
    (cd "${FORK_DIR}" && pixi run python train.py \
      --pretrain_ckpt model_checkpoints/Kd+Ki+IC50_experimental_fine_tuning.ckpt \
      --train_set_path "${REPO_DIR}/structures/gator/folds/fold${K}_train.pkl" \
      --valid_set_path "${REPO_DIR}/structures/gator/folds/fold${K}_val.pkl" \
      --save_dir "${SAVE_DIR}" \
      --partial_finetune True \
      --lr "${LR_INIT}" --warmup_end_lr "${LR_WARM}" --cos_lr "${LR_COS}" \
      --batch_size "${BATCH_SIZE}" --valid_batch_size 48 \
      --max_epoch "${MAX_EPOCH}" --warmup_epochs "${WARMUP}" \
      --patience "${PATIENCE}" --save_topk 2 --seed 42) > "${LOG}" 2>&1
  fi

  echo "[fold${K}] inference on val pkl"
  pixi --manifest-path "${REPO_DIR}/pyproject.toml" run python \
    "${REPO_DIR}/track1_activity/scripts/gator/04_infer_fold.py" \
    --fold "${K}" --save-dir "${SAVE_DIR}"
done

echo
echo "Done. Next: pixi run python track1_activity/scripts/gator/06_collate_oof.py --run ${RUN_NAME}"
