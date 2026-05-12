#!/usr/bin/env bash
# Upgrade the 8k fast Boltz-2 trunk embeddings from recycling_steps=1 to
# recycling_steps=3 by resuming from the existing rcycle=1 s/z tensors.
#
# Source embeddings:
#   structures/boltz2/outputs_fast/boltz_results_inputs_fast/predictions/<id>/embeddings_<id>.npz
#
# Target embeddings:
#   structures/boltz2/outputs_fast_rcycle3/boltz_results_inputs_fast/predictions/<id>/embeddings_<id>.npz
#
# The script stages only compounds with an existing source embedding and no
# target embedding yet, so it is safe to re-launch after interruption. It does
# not run diffusion, confidence, or affinity; it only advances the trunk recycle
# state and writes the upgraded embeddings.
#
# To avoid changing token/atom order, the script reuses the original fast run's
# processed cache. It symlinks only the staged records into the target processed
# tree; symlinking the whole processed directory would make Boltz load the full
# cached manifest and re-run already completed records.
#
# Requires the N283T community fork with --resume_embeddings_dir support, e.g.
#   uv tool install --python 3.12 --reinstall --force \
#       "git+https://github.com/N283T/boltz-community.git[cuda]"
#
# Typical RTX 5080 run:
#   bash track1_activity/scripts/boltz_affhead/boltz2_fast_resume_rcycle3_embeddings_run.sh
#
# Mac / MPS smoke or distributed shard example:
#   BOLTZ_ACCELERATOR=mps BOLTZ_DEVICES=1 \
#     bash track1_activity/scripts/boltz_affhead/boltz2_fast_resume_rcycle3_embeddings_run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BOLTZ_BIN="${BOLTZ_BIN:-}"
if [[ -z "$BOLTZ_BIN" ]]; then
    if command -v boltz >/dev/null 2>&1; then
        BOLTZ_BIN="$(command -v boltz)"
    elif [[ -x "$HOME/.local/bin/boltz" ]]; then
        BOLTZ_BIN="$HOME/.local/bin/boltz"
    else
        echo "ERROR: boltz CLI not found. Install the community fork first:" >&2
        echo "  uv tool install --python 3.12 --reinstall --force \\" >&2
        echo "      \"git+https://github.com/N283T/boltz-community.git[cuda]\"" >&2
        exit 1
    fi
fi

# torch/cuequivariance wheels can place CUDA runtime libraries in package-local
# directories. Add all known nvidia lib directories so Triton/NVRTC dlopen works.
for NVIDIA_SITE in \
    "$HOME/.local/share/uv/tools/boltz/lib/python3.12/site-packages/nvidia" \
    "$HOME/.local/share/uv/tools/boltz-community/lib/python3.12/site-packages/nvidia"
do
    if [[ -d "$NVIDIA_SITE" ]]; then
        while IFS= read -r lib_dir; do
            export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
        done < <(find "$NVIDIA_SITE" -maxdepth 3 -type d -name lib | sort -r)
    fi
done

if ! "$BOLTZ_BIN" predict --help 2>&1 | grep -q -- "--resume_embeddings_dir"; then
    echo "ERROR: $BOLTZ_BIN does not expose --resume_embeddings_dir." >&2
    echo "       Reinstall N283T/boltz-community or the local community clone." >&2
    exit 1
fi
if ! "$BOLTZ_BIN" predict --help 2>&1 | grep -q -- "--embeddings_only"; then
    echo "ERROR: $BOLTZ_BIN does not expose --embeddings_only." >&2
    exit 1
fi

INPUTS_DIR="$REPO_ROOT/structures/boltz2/inputs_fast"
RESUME_DIR="$REPO_ROOT/structures/boltz2/outputs_fast/boltz_results_inputs_fast/predictions"
OUTPUT_DIR="$REPO_ROOT/structures/boltz2/outputs_fast_rcycle3"
RESULTS_NAME="boltz_results_inputs_fast"
SOURCE_RESULTS_DIR="$REPO_ROOT/structures/boltz2/outputs_fast/$RESULTS_NAME"
TARGET_RESULTS_DIR="$OUTPUT_DIR/$RESULTS_NAME"
TARGET_PRED_DIR="$TARGET_RESULTS_DIR/predictions"
LOG_DIR="$REPO_ROOT/logs"

SEED="${BOLTZ_SEED:-42}"
SOURCE_RCYCLE="${BOLTZ_SOURCE_RCYCLE:-1}"
TARGET_RCYCLE="${BOLTZ_TARGET_RCYCLE:-3}"
ACCELERATOR="${BOLTZ_ACCELERATOR:-gpu}"
DEVICES="${BOLTZ_DEVICES:-1}"
NUM_WORKERS="${BOLTZ_NUM_WORKERS:-0}"
PREPROCESSING_THREADS="${BOLTZ_PREPROCESSING_THREADS:-32}"
LIMIT="${BOLTZ_LIMIT:-0}"
REUSE_PROCESSED="${BOLTZ_REUSE_PROCESSED:-1}"

if [[ ! -d "$INPUTS_DIR" ]]; then
    echo "ERROR: inputs dir not found: $INPUTS_DIR" >&2
    exit 1
fi
if [[ ! -d "$RESUME_DIR" ]]; then
    echo "ERROR: resume embeddings dir not found: $RESUME_DIR" >&2
    exit 1
fi
if [[ ! -d "$SOURCE_RESULTS_DIR/processed" ]]; then
    echo "ERROR: source processed cache not found: $SOURCE_RESULTS_DIR/processed" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$TARGET_RESULTS_DIR" "$LOG_DIR"
LOG="$LOG_DIR/boltz2_fast_resume_rcycle3_embeddings_$(date +%Y%m%d_%H%M%S).log"

SCRATCH_DIR="$(mktemp -d -t boltz2_fast_r3.XXXXXX)"
SCRATCH_INPUT="$SCRATCH_DIR/inputs_fast"
mkdir -p "$SCRATCH_INPUT"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

staged_ids=()
staged=0
missing_source=0
already_done=0
for yaml in "$INPUTS_DIR"/*.yaml; do
    [[ -e "$yaml" ]] || continue
    cid="$(basename "$yaml" .yaml)"
    source_npz="$RESUME_DIR/$cid/embeddings_$cid.npz"
    target_npz="$TARGET_PRED_DIR/$cid/embeddings_$cid.npz"
    if [[ ! -f "$source_npz" ]]; then
        missing_source=$((missing_source + 1))
        continue
    fi
    if [[ -f "$target_npz" ]]; then
        already_done=$((already_done + 1))
        continue
    fi
    ln -sf "$(realpath "$yaml")" "$SCRATCH_INPUT/$cid.yaml"
    staged_ids+=("$cid")
    staged=$((staged + 1))
    if [[ "$LIMIT" -gt 0 && "$staged" -ge "$LIMIT" ]]; then
        break
    fi
done

link_one() {
    local source_file="$1"
    local target_file="$2"
    if [[ ! -e "$source_file" ]]; then
        return
    fi
    if [[ -e "$target_file" || -L "$target_file" ]]; then
        return
    fi
    ln -s "$source_file" "$target_file"
}

prepare_reused_processed_cache() {
    local target_processed="$TARGET_RESULTS_DIR/processed"

    # Rebuild this cache each launch so stale records cannot make Boltz load the
    # full previous manifest. This does not remove predictions.
    rm -rf "$target_processed" "$TARGET_RESULTS_DIR/msa"

    mkdir -p "$target_processed"/{records,structures,constraints,mols,msa,templates}
    mkdir -p "$TARGET_RESULTS_DIR/msa"

    for cid in "${staged_ids[@]}"; do
        link_one "$SOURCE_RESULTS_DIR/processed/records/$cid.json" \
            "$target_processed/records/$cid.json"
        link_one "$SOURCE_RESULTS_DIR/processed/structures/$cid.npz" \
            "$target_processed/structures/$cid.npz"
        link_one "$SOURCE_RESULTS_DIR/processed/constraints/$cid.npz" \
            "$target_processed/constraints/$cid.npz"
        link_one "$SOURCE_RESULTS_DIR/processed/mols/$cid.pkl" \
            "$target_processed/mols/$cid.pkl"
        for msa_npz in "$SOURCE_RESULTS_DIR/processed/msa/$cid"_*.npz; do
            [[ -e "$msa_npz" ]] || continue
            link_one "$msa_npz" "$target_processed/msa/$(basename "$msa_npz")"
        done
    done
}

{
    echo "=== Boltz fast rcycle upgrade ==="
    date
    echo "boltz:          $("$BOLTZ_BIN" --help 2>&1 | head -1)"
    echo "boltz_bin:      $BOLTZ_BIN"
    echo "inputs:         $INPUTS_DIR"
    echo "resume_dir:     $RESUME_DIR"
    echo "outputs:        $OUTPUT_DIR"
    echo "source_cache:   $SOURCE_RESULTS_DIR/processed"
    echo "target_cache:   $TARGET_RESULTS_DIR/processed"
    echo "source_rcycle:  $SOURCE_RCYCLE"
    echo "target_rcycle:  $TARGET_RCYCLE"
    echo "seed:           $SEED"
    echo "accelerator:    $ACCELERATOR"
    echo "devices:        $DEVICES"
    echo "num_workers:    $NUM_WORKERS"
    echo "preprocess_thr: $PREPROCESSING_THREADS"
    echo "reuse_processed:$REUSE_PROCESSED"
    echo "limit:          $LIMIT"
    echo "staged:         $staged"
    echo "already_done:   $already_done"
    echo "missing_source: $missing_source"
    echo "log:            $LOG"
    echo ""
} | tee "$LOG"

if [[ "$staged" -eq 0 ]]; then
    echo "Nothing to run." | tee -a "$LOG"
    exit 0
fi
if [[ "$REUSE_PROCESSED" == "1" ]]; then
    prepare_reused_processed_cache
fi
if [[ "${BOLTZ_DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run only. No Boltz command executed." | tee -a "$LOG"
    exit 0
fi

extra_args=()
if [[ "${BOLTZ_NO_KERNELS:-0}" == "1" ]]; then
    extra_args+=(--no_kernels)
fi
if [[ -n "${BOLTZ_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    user_extra_args=($BOLTZ_EXTRA_ARGS)
    extra_args+=("${user_extra_args[@]}")
fi

"$BOLTZ_BIN" predict "$SCRATCH_INPUT" \
    --out_dir "$OUTPUT_DIR" \
    --model boltz2 \
    --embeddings_only \
    --resume_embeddings_dir "$RESUME_DIR" \
    --resume_recycling_steps "$SOURCE_RCYCLE" \
    --recycling_steps "$TARGET_RCYCLE" \
    --seed "$SEED" \
    --output_format mmcif \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --num_workers "$NUM_WORKERS" \
    --preprocessing-threads "$PREPROCESSING_THREADS" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
find "$TARGET_PRED_DIR" -name "embeddings_*.npz" 2>/dev/null \
    | wc -l \
    | awk '{print "target embeddings_*.npz: " $1}' \
    | tee -a "$LOG"
