#!/usr/bin/env bash
# Smoke test: run boltz predict on the 10-compound smoke set with R1 settings.
# R1 = use_potentials ON, diffusion_samples=1, recycling_steps=3 (default).
#
# Outputs land in structures/boltz2/outputs_smoke/. Logs go to logs/boltz2_smoke.log.
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

INPUT_DIR="structures/boltz2/inputs_smoke"
OUTPUT_DIR="structures/boltz2/outputs_smoke"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/boltz2_smoke.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[boltz2_smoke] start: $(date -Is)" | tee "$LOG_FILE"
echo "[boltz2_smoke] input_dir : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_smoke] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_smoke] boltz: $(boltz --help 2>&1 | head -1)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# R1 settings
boltz predict "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --use_potentials \
    --diffusion_samples 1 \
    --recycling_steps 3 \
    --output_format mmcif \
    --accelerator gpu \
    --devices 1 \
    --num_workers 2 \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[boltz2_smoke] done: $(date -Is)" | tee -a "$LOG_FILE"
