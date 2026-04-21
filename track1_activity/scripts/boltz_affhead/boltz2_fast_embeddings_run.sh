#!/usr/bin/env bash
# Fast trunk embedding extraction for the 8,483 compounds not covered by
# the main compound_boltz2 table. Uses the user's Boltz fork's
# --embeddings_only flag + --recycling_steps 1 to minimize compute: no
# diffusion sampling, no confidence head, no affinity head.
#
# Seed is pinned to 42 per project convention (CLAUDE.md "Pin seed for
# long runs" memory note) — even though trunk-only at rcycle=1 is mostly
# deterministic, pinning eliminates any residual nondeterminism (dropout
# defaults, tie-breaking on exact tensor values) and matches how our
# existing rcycle=3 run was configured.
#
# Outputs: structures/boltz2/outputs_fast/boltz_results_inputs_fast/
#     predictions/<id>/embeddings_<id>.npz
#
# Requires the patched boltz fork installed via uv tool:
#   uv tool install --python 3.12 --reinstall --force --editable \\
#       "$HOME/ghq/github.com/N283T/boltz[cuda]"
#
# Resume-safe: --embeddings_only flag makes filter_inputs_structure
# treat a per-compound folder as "done" only when embeddings_<id>.npz
# exists, so an interrupted run can be re-launched safely.
#
# Typical invocation (in tmux for detachable long run):
#   tmux new -s boltz_fast
#   bash track1_activity/scripts/boltz_affhead/boltz2_fast_embeddings_run.sh
#   # Ctrl-b d to detach; tmux attach -t boltz_fast to reattach.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# torch 2.11+cu130 inside the boltz uv tool venv ships CUDA NVRTC libs
# under nvidia/cu13/lib/. dlopen needs this on path for JIT triton/
# cuequivariance kernels to resolve libnvrtc-builtins.so.13.0.
NVIDIA_CU13_LIB_DIR="$HOME/.local/share/uv/tools/boltz/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$NVIDIA_CU13_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

INPUTS_DIR="$REPO_ROOT/structures/boltz2/inputs_fast"
OUTPUT_DIR="$REPO_ROOT/structures/boltz2/outputs_fast"
LOG_DIR="$REPO_ROOT/logs"

if [[ ! -d "$INPUTS_DIR" ]]; then
    echo "ERROR: inputs dir not found: $INPUTS_DIR" >&2
    echo "Run track1_activity/scripts/boltz_affhead/boltz2_build_inputs_fast.py first." >&2
    exit 1
fi

if ! command -v boltz >/dev/null 2>&1; then
    echo "ERROR: boltz CLI not found. Install via:" >&2
    echo "  uv tool install --python 3.12 --editable \"\$HOME/ghq/github.com/N283T/boltz[cuda]\"" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG="$LOG_DIR/boltz2_fast_embeddings_$(date +%Y%m%d_%H%M%S).log"

SEED="${BOLTZ_SEED:-42}"
RCYCLE="${BOLTZ_RCYCLE:-1}"

{
    echo "=== Fast trunk run ==="
    date
    echo "inputs:         $INPUTS_DIR  ($(ls "$INPUTS_DIR" | wc -l) YAMLs)"
    echo "outputs:        $OUTPUT_DIR"
    echo "seed:           $SEED"
    echo "rcycle:         $RCYCLE"
    echo "mode:           --embeddings_only (trunk only, no diffusion/affinity)"
    echo "log:            $LOG"
    echo ""

    boltz predict "$INPUTS_DIR" \
        --embeddings_only \
        --recycling_steps "$RCYCLE" \
        --seed "$SEED" \
        --out_dir "$OUTPUT_DIR" \
        --accelerator gpu

    echo ""
    echo "=== Done at $(date) ==="
    # Quick sanity count
    n=$(find "$OUTPUT_DIR" -name "embeddings_*.npz" 2>/dev/null | wc -l)
    echo "embeddings_*.npz produced: $n"
} 2>&1 | tee "$LOG"
