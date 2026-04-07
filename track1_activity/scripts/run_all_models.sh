#!/usr/bin/env bash
# Run all model training sequentially (GPU-bound, one at a time)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== [$(date)] Starting model training pipeline ==="

echo ""
echo "=== [$(date)] 1/3: ChemProp Optuna ==="
pixi run python track1_activity/scripts/run_chemprop_optuna.py --n-trials 50 --split umap

echo ""
echo "=== [$(date)] 2/3: AttentiveFP Optuna ==="
pixi run python track1_activity/scripts/run_attentivefp_optuna.py --n-trials 50 --split umap

echo ""
echo "=== [$(date)] 3/3: MoLFormer Fine-tune ==="
pixi run python track1_activity/scripts/run_molformer_finetune.py --n-trials 30 --split umap

echo ""
echo "=== [$(date)] All done! ==="
