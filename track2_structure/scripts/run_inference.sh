#!/usr/bin/env bash
# Track 2 main run: Boltz-2 prediction for all 184 holo PXR-ligand YAMLs.
#
# Settings reflect agreed-upon Track 2 baseline (2026-04-25):
#   --diffusion_samples 5    : best-of-N pose generation
#   --recycling_steps 5      : Boltz-2 paper recommended setting
#   --sampling_steps 200     : default
#   --step_scale 1.5         : default
#   --use_potentials         : essential for chirality and steric correctness
#   --use_msa_server         : ColabFold MSA (cached after the apo run)
#   --write_embeddings       : dump s/z trunk embeddings (reusable for Track 1)
#   --output_format pdb      : direct PDB output for submission packaging
#   --seed 42                : fixed for reproducibility
#
# Boltz CLI auto-resumes by skipping compound directories that already contain
# a prediction, so this script is safe to interrupt and re-launch.
#
# Usage:
#   tmux new -s track2_boltz2
#   bash track2_structure/scripts/run_inference.sh
#
# Optional smoke test (subset of YAMLs):
#   INPUT_DIR=track2_structure/inputs/holo_smoke bash run_inference.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# torch 2.11+cu130 inside the boltz uv tool venv ships its CUDA NVRTC libs
# under nvidia/cu13/lib/. dlopen does not look there by default, so the JIT
# kernel compilation in triton/cuequivariance fails with
# "failed to open libnvrtc-builtins.so.13.0". Adding the directory to
# LD_LIBRARY_PATH lets dlopen resolve it.
NVIDIA_CU13_LIB_DIR="$HOME/.local/share/uv/tools/boltz/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$NVIDIA_CU13_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

INPUT_DIR="${INPUT_DIR:-track2_structure/inputs/holo}"
OUTPUT_DIR="${OUTPUT_DIR:-structures/boltz2_track2/outputs/holo}"
LOG_DIR="logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/boltz2_track2_holo.log}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if ! ls "$INPUT_DIR"/*.yaml >/dev/null 2>&1; then
    echo "[track2] no YAMLs found in $INPUT_DIR — run build_inputs.py --mode holo first." >&2
    exit 1
fi

YAML_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' | wc -l)

echo "[track2_holo] start: $(date -Is)" | tee "$LOG_FILE"
echo "[track2_holo] input_dir : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[track2_holo] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[track2_holo] yaml count: $YAML_COUNT" | tee -a "$LOG_FILE"
echo "[track2_holo] boltz: $(boltz --help 2>&1 | head -1)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

boltz predict "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --use_msa_server \
    --use_potentials \
    --diffusion_samples 5 \
    --recycling_steps 5 \
    --sampling_steps 200 \
    --step_scale 1.5 \
    --write_embeddings \
    --output_format pdb \
    --seed 42 \
    --accelerator gpu \
    --devices 1 \
    --num_workers 2 \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[track2_holo] done: $(date -Is)" | tee -a "$LOG_FILE"
