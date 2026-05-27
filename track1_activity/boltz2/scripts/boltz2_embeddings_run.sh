#!/usr/bin/env bash
# Embeddings-only run: extract trunk s/z embeddings for all 4653 PXR
# compounds without re-running diffusion sampling, confidence, or
# affinity. Reuses the cached preprocessed records and MSA NPZs from
# the prior full structure run, so no MSA download or preprocessing
# is repeated.
#
# Outputs land in structures/boltz2/outputs/boltz_results_inputs/predictions/<id>/
# as ``embeddings_<id>.npz`` (s + z arrays), placed alongside the
# existing CIF/JSON/PAE/PDE/affinity files. Existing files are not
# modified (verified by smoke test on 2026-04-16).
#
# Requires the patched official Boltz checkout:
#   /home/nagaet/ghq/github.com/jwohlwend/boltz
# on branch codex/resume-embeddings-from-trunk.
#
# Estimated runtime on RTX 5080: ~17 hours
# Estimated disk usage: ~250 GB (embeddings_*.npz, ~55 MB / compound)
#
# The run is resume-safe: ``--embeddings_only`` makes
# filter_inputs_structure treat a per-compound folder as "already done"
# only when ``embeddings_<id>.npz`` is present, so an interrupted run
# can be re-launched with the same command.
#
# Logs go to logs/boltz2_embeddings.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# torch 2.11+cu130 inside the Boltz uv venv ships its CUDA NVRTC libs
# under nvidia/cu13/lib/. dlopen does not look there by default, so the JIT
# kernel compilation in triton/cuequivariance fails with
# "failed to open libnvrtc-builtins.so.13.0". Adding the directory to
# LD_LIBRARY_PATH lets dlopen resolve it.
BOLTZ_PROJECT="${BOLTZ_PROJECT:-$HOME/ghq/github.com/jwohlwend/boltz}"
BOLTZ_UV_EXTRA="${BOLTZ_UV_EXTRA:-cuda}"
NVIDIA_CU13_LIB_DIR="${BOLTZ_PROJECT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$NVIDIA_CU13_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

INPUT_DIR="structures/boltz2/inputs"
OUTPUT_DIR="structures/boltz2/outputs"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/boltz2_embeddings.log"
NUM_WORKERS="${NUM_WORKERS:-2}"
NO_KERNELS="${NO_KERNELS:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -d "$BOLTZ_PROJECT/.git" ]]; then
    echo "ERROR: official Boltz checkout not found: $BOLTZ_PROJECT" >&2
    exit 1
fi

# Sanity: make sure the patched official checkout is what we run.
uv_args=(--project "$BOLTZ_PROJECT")
if [[ -n "$BOLTZ_UV_EXTRA" ]]; then
    uv_args+=(--extra "$BOLTZ_UV_EXTRA")
fi

BOLTZ_VERSION_LINE="$(uv run "${uv_args[@]}" boltz --help 2>&1 | head -1)"
if ! uv run "${uv_args[@]}" boltz predict --help 2>&1 \
    | grep -q -- '--embeddings_only'; then
    echo "ERROR: $BOLTZ_PROJECT does not expose --embeddings_only." >&2
    echo "       Switch to / install branch codex/resume-embeddings-from-trunk." >&2
    exit 1
fi
kernel_args=()
if [[ "$NO_KERNELS" == "1" ]]; then
    kernel_args+=(--no_kernels)
fi

echo "[boltz2_embeddings] start: $(date -Is)" | tee "$LOG_FILE"
echo "[boltz2_embeddings] input_dir : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] boltz_project: $BOLTZ_PROJECT" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] boltz_uv_extra: ${BOLTZ_UV_EXTRA:-<none>}" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] yaml count: $(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' | wc -l)" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] boltz: $BOLTZ_VERSION_LINE" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] seed=42 (pinned for reproducibility, see issue #57)" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] num_workers=$NUM_WORKERS" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] no_kernels=$NO_KERNELS" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# R1 settings minus diffusion / confidence / affinity (skipped by
# --embeddings_only). recycling_steps and seed are kept so the trunk
# pass produces a canonical, reproducible (s, z) tensor pair.
uv run "${uv_args[@]}" boltz predict "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --embeddings_only \
    --recycling_steps 3 \
    --seed 42 \
    --output_format mmcif \
    --accelerator gpu \
    --devices 1 \
    --num_workers "$NUM_WORKERS" \
    "${kernel_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] done: $(date -Is)" | tee -a "$LOG_FILE"
