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
# Requires the patched boltz fork installed via:
#   uv tool install --python 3.12 --reinstall --force --editable \
#       '/home/nagaet/ghq/github.com/N283T/boltz[cuda]'
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

INPUT_DIR="structures/boltz2/inputs"
OUTPUT_DIR="structures/boltz2/outputs"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/boltz2_embeddings.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Sanity: make sure the patched fork is what is on PATH.
BOLTZ_VERSION_LINE="$(boltz --help 2>&1 | head -1)"
if ! boltz predict --help 2>&1 | grep -q -- '--embeddings_only'; then
    echo "ERROR: the installed boltz binary does not expose --embeddings_only." >&2
    echo "       reinstall the patched fork with:" >&2
    echo "       uv tool install --python 3.12 --reinstall --force --editable" >&2
    echo "         '/home/nagaet/ghq/github.com/N283T/boltz[cuda]'" >&2
    exit 1
fi

echo "[boltz2_embeddings] start: $(date -Is)" | tee "$LOG_FILE"
echo "[boltz2_embeddings] input_dir : $INPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] yaml count: $(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' | wc -l)" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] boltz: $BOLTZ_VERSION_LINE" | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] seed=42 (pinned for reproducibility, see issue #57)" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# R1 settings minus diffusion / confidence / affinity (skipped by
# --embeddings_only). recycling_steps and seed are kept so the trunk
# pass produces a canonical, reproducible (s, z) tensor pair.
boltz predict "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --embeddings_only \
    --recycling_steps 3 \
    --seed 42 \
    --output_format mmcif \
    --accelerator gpu \
    --devices 1 \
    --num_workers 2 \
    2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "[boltz2_embeddings] done: $(date -Is)" | tee -a "$LOG_FILE"
