#!/usr/bin/env bash
# Apo (ligand-free) Boltz-2 run for the LBD-only PXR sequence used by Track 2.
#
# Goals:
#   1. Trigger ColabFold MSA fetch + cache the result so subsequent 184-compound
#      runs reuse the same MSA.
#   2. Produce an apo LBD model for sanity checking / analysis (compare to
#      our existing holo crystals in structures/pxr_lbd/).
#   3. Validate the LBD-only sequence works with the current Boltz-2 install.
#
# Settings are intentionally light: the apo run is a setup step, not the
# scientific output. Default samples=1 / recycling=3 is plenty.
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

INPUT_DIR="track2_structure/inputs/apo"
OUTPUT_DIR="structures/boltz2_track2/outputs/apo"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/boltz2_track2_apo.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -f "$INPUT_DIR/apo.yaml" ]]; then
    echo "[apo] $INPUT_DIR/apo.yaml not found — run build_inputs.py --mode apo first." >&2
    exit 1
fi

echo "[track2_apo] start: $(date -Is)" | tee "$LOG_FILE"
echo "[track2_apo] input_dir : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[track2_apo] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[track2_apo] boltz: $(boltz --help 2>&1 | head -1)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# Light settings; this run exists to populate the MSA cache and sanity-check
# the LBD-only sequence. ``--use_potentials`` is on for consistency with the
# main run (ensures any kernel/init issues surface here, not later).
boltz predict "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --use_msa_server \
    --use_potentials \
    --diffusion_samples 1 \
    --recycling_steps 3 \
    --output_format pdb \
    --seed 42 \
    --accelerator gpu \
    --devices 1 \
    --num_workers 2 \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[track2_apo] done: $(date -Is)" | tee -a "$LOG_FILE"
