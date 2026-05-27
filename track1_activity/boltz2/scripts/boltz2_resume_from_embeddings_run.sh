#!/usr/bin/env bash
# Resume Boltz-2 structure prediction from previously saved trunk embeddings.
#
# This uses the patched official Boltz checkout, not boltz-community and not
# the older N283T embeddings-only fork. The resume path reads
# embeddings_<id>.npz files containing official-Boltz s/z tensors, skips the
# trunk rounds represented by --resume_recycling_steps, then runs diffusion
# and confidence. Affinity is skipped by default for structure-only recovery;
# set SKIP_AFFINITY=0 to reuse the saved embeddings in the affinity crop too.
#
# Default output is separate from the production tree so the regenerated
# structures can be inspected before replacing anything:
#   structures/boltz2/outputs_resume_from_embeddings/boltz_results_inputs/
#
# To overwrite the main output tree from the existing embeddings, set:
#   OUTPUT_DIR=structures/boltz2/outputs OVERRIDE=1 bash ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BOLTZ_PROJECT="${BOLTZ_PROJECT:-$HOME/ghq/github.com/jwohlwend/boltz}"
BOLTZ_UV_EXTRA="${BOLTZ_UV_EXTRA:-cuda}"
INPUT_DIR="${INPUT_DIR:-structures/boltz2/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-structures/boltz2/outputs_resume_from_embeddings}"
RESUME_EMBEDDINGS_DIR="${RESUME_EMBEDDINGS_DIR:-structures/boltz2/outputs/boltz_results_inputs/predictions}"
PROCESSED_DIR="${PROCESSED_DIR:-structures/boltz2/outputs/boltz_results_inputs/processed}"
RESUME_RECYCLING_STEPS="${RESUME_RECYCLING_STEPS:-3}"
RECYCLING_STEPS="${RECYCLING_STEPS:-3}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NO_KERNELS="${NO_KERNELS:-0}"
SKIP_AFFINITY="${SKIP_AFFINITY:-1}"
WRITE_AFFINITY_EMBEDDINGS="${WRITE_AFFINITY_EMBEDDINGS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/boltz2_resume_from_embeddings.log}"
OVERRIDE="${OVERRIDE:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -d "$BOLTZ_PROJECT/.git" ]]; then
    echo "ERROR: official Boltz checkout not found: $BOLTZ_PROJECT" >&2
    exit 1
fi

uv_args=(--project "$BOLTZ_PROJECT")
if [[ -n "$BOLTZ_UV_EXTRA" ]]; then
    uv_args+=(--extra "$BOLTZ_UV_EXTRA")
fi

if ! uv run "${uv_args[@]}" boltz predict --help 2>&1 \
    | grep -q -- '--resume_embeddings_dir'; then
    echo "ERROR: $BOLTZ_PROJECT does not expose --resume_embeddings_dir." >&2
    echo "       Switch to / install branch codex/resume-embeddings-from-trunk." >&2
    exit 1
fi

override_args=()
if [[ "$OVERRIDE" == "1" ]]; then
    override_args+=(--override)
fi
kernel_args=()
if [[ "$NO_KERNELS" == "1" ]]; then
    kernel_args+=(--no_kernels)
fi
affinity_args=()
if [[ "$SKIP_AFFINITY" == "1" ]]; then
    affinity_args+=(--skip_affinity)
fi
if [[ "$WRITE_AFFINITY_EMBEDDINGS" == "1" ]]; then
    affinity_args+=(--write_affinity_embeddings)
fi

echo "[boltz2_resume_from_embeddings] start: $(date -Is)" | tee "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] boltz_project : $BOLTZ_PROJECT" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] boltz_uv_extra: ${BOLTZ_UV_EXTRA:-<none>}" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] input_dir     : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] output_dir    : $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] embeddings_dir: $RESUME_EMBEDDINGS_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] processed_dir : $PROCESSED_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] resume_recycling_steps=$RESUME_RECYCLING_STEPS recycling_steps=$RECYCLING_STEPS seed=$SEED num_workers=$NUM_WORKERS no_kernels=$NO_KERNELS skip_affinity=$SKIP_AFFINITY write_affinity_embeddings=$WRITE_AFFINITY_EMBEDDINGS" | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] yaml count: $(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' | wc -l)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

if [[ -n "$PROCESSED_DIR" ]]; then
    if [[ ! -d "$PROCESSED_DIR" ]]; then
        echo "ERROR: processed dir not found: $PROCESSED_DIR" >&2
        exit 1
    fi
    result_dir="$OUTPUT_DIR/boltz_results_$(basename "$INPUT_DIR")"
    mkdir -p "$result_dir"
    if [[ ! -e "$result_dir/processed" ]]; then
        ln -s "$(realpath "$PROCESSED_DIR")" "$result_dir/processed"
    fi
fi

uv run "${uv_args[@]}" boltz predict "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --resume_embeddings_dir "$RESUME_EMBEDDINGS_DIR" \
    --resume_recycling_steps "$RESUME_RECYCLING_STEPS" \
    --use_potentials \
    --diffusion_samples 1 \
    --recycling_steps "$RECYCLING_STEPS" \
    --output_format mmcif \
    --write_embeddings \
    --seed "$SEED" \
    --accelerator gpu \
    --devices 1 \
    --num_workers "$NUM_WORKERS" \
    "${kernel_args[@]}" \
    "${affinity_args[@]}" \
    "${override_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[boltz2_resume_from_embeddings] done: $(date -Is)" | tee -a "$LOG_FILE"
