#!/usr/bin/env bash
# Re-run Boltz-2 inference for a specific list of compound IDs and merge the
# predictions into the main output tree (structures/boltz2/outputs/), so
# boltz2_postprocess.py picks them up without any path override.
#
# The script stages the requested YAMLs into a scratch directory whose
# basename is "inputs" (matching the full run), so boltz writes to
# structures/boltz2/outputs/boltz_results_inputs/predictions/<id>/ — exactly
# where the full-run artifacts live. Stale predictions and cached
# preprocessing NPZs for the target compounds are removed first so boltz
# re-runs them from scratch (otherwise it resume-skips by directory
# presence).
#
# Usage:
#     bash track2_structure/scripts/boltz2_recover_run.sh <compound_id> [<compound_id> ...]
# Example:
#     bash track2_structure/scripts/boltz2_recover_run.sh 03840
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <compound_id> [<compound_id> ...]" >&2
    exit 2
fi

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

INPUT_MAIN="structures/boltz2/inputs"
OUTPUT_DIR="structures/boltz2/outputs"
RESULTS_DIR="${OUTPUT_DIR}/boltz_results_inputs"
PREDICTIONS_DIR="${RESULTS_DIR}/predictions"
PROCESSED_DIR="${RESULTS_DIR}/processed"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/boltz2_recover_$(date +%Y%m%d_%H%M%S).log"

SCRATCH_DIR="$(mktemp -d -t boltz2_recover.XXXXXX)"
SCRATCH_INPUT="${SCRATCH_DIR}/inputs"
mkdir -p "$SCRATCH_INPUT" "$OUTPUT_DIR" "$LOG_DIR"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

echo "[boltz2_recover] start: $(date -Is)" | tee "$LOG_FILE"

for arg in "$@"; do
    if ! [[ "$arg" =~ ^[0-9]+$ ]]; then
        echo "ERROR: compound id must be numeric: $arg" >&2
        exit 2
    fi
    padded=$(printf "%05d" "$arg")
    yaml="${INPUT_MAIN}/${padded}.yaml"
    if [[ ! -f "$yaml" ]]; then
        echo "ERROR: input YAML missing: $yaml" >&2
        exit 2
    fi
    ln -sf "$(realpath "$yaml")" "${SCRATCH_INPUT}/${padded}.yaml"

    # Force fresh prediction: drop existing output and per-compound cache.
    rm -rf "${PREDICTIONS_DIR}/${padded}"
    for subdir in structures constraints mols records msa; do
        if [[ -d "${PROCESSED_DIR}/${subdir}" ]]; then
            find "${PROCESSED_DIR}/${subdir}" -maxdepth 1 \
                -name "${padded}*" -delete
        fi
    done
    echo "[boltz2_recover] staged id=${padded} yaml=${yaml}" | tee -a "$LOG_FILE"
done

yaml_count=$(find "$SCRATCH_INPUT" -maxdepth 1 -name '*.yaml' | wc -l)
echo "[boltz2_recover] input_dir : $SCRATCH_INPUT" | tee -a "$LOG_FILE"
echo "[boltz2_recover] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_recover] yaml count: $yaml_count" | tee -a "$LOG_FILE"
echo "[boltz2_recover] boltz: $(boltz --help 2>&1 | head -1)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# R1 settings (match boltz2_full_run.sh)
boltz predict "$SCRATCH_INPUT" \
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
echo "[boltz2_recover] done: $(date -Is)" | tee -a "$LOG_FILE"
echo "[boltz2_recover] next: pixi run python track2_structure/scripts/boltz2_postprocess.py --db" \
    | tee -a "$LOG_FILE"
