#!/bin/bash
# Scheduled LB submission — polls api.py cooldown and submits once READY.
#
# Designed to run unattended via nohup + disown so it survives terminal/session
# closes. Logs to a timestamped file under logs/ for post-hoc audit.
#
# Usage:
#   scheduled_submit.sh <csv_path> \
#       --experiment <experiment_name> \
#       [--notes <notes_text>] \
#       [--track <track_name>] \
#       [--poll <seconds>] \
#       [--safety <seconds>] \
#       [--log <log_path>]
#
# Defaults:
#   --track   "Activity Prediction"
#   --poll    180   (check cooldown every 3 minutes)
#   --safety  60    (sleep this long after cooldown READY before submitting,
#                    in case of clock skew with LB server)
#   --log     logs/scheduled_submit_<timestamp>.log
#
# Typical background invocation:
#   nohup bash track1_activity/scripts/scheduled_submit.sh \
#       track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv \
#       --experiment ens_caruana_bag20_calibrated_best \
#       --notes "blah blah" \
#       > /tmp/scheduled_submit_status.log 2>&1 &
#   disown
#
# Exit codes:
#   0  submission successful
#   1  usage / argument error
#   2  CSV not found
#   3  submission failed (api.py exit non-zero)
set -euo pipefail

CSV=""
EXP=""
NOTES=""
TRACK="Activity Prediction"
POLL=180
SAFETY=60
LOG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXP="$2"; shift 2 ;;
        --notes) NOTES="$2"; shift 2 ;;
        --track) TRACK="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        --safety) SAFETY="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#'
            exit 0
            ;;
        *)
            if [[ -z "$CSV" ]]; then
                CSV="$1"
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CSV" ]] || [[ -z "$EXP" ]]; then
    echo "Usage: $0 <csv_path> --experiment <name> [--notes <text>] [--track <track>] [--poll N] [--safety N] [--log PATH]" >&2
    exit 1
fi

if [[ ! -f "$CSV" ]]; then
    echo "ERROR: CSV not found: $CSV" >&2
    exit 2
fi

# Locate repo root (script is at <repo>/track1_activity/scripts/scheduled_submit.sh)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ -z "$LOG" ]]; then
    mkdir -p logs
    LOG="logs/scheduled_submit_$(date +%Y%m%d_%H%M%S).log"
fi

{
    echo "=== scheduled_submit ==="
    date
    echo "  CSV:        $CSV"
    echo "  Experiment: $EXP"
    echo "  Track:      $TRACK"
    echo "  Notes:      ${NOTES:-(none)}"
    echo "  Poll:       ${POLL}s"
    echo "  Safety:     ${SAFETY}s"
    echo "  Log:        $LOG"
    echo "  Repo:       $REPO"
    echo ""

    while true; do
        SET_PLUS_E=0
        set +e
        OUTPUT="$(pixi run python track1_activity/scripts/api.py cooldown 2>&1)"
        RC=$?
        set -e
        if [[ $RC -ne 0 ]]; then
            echo "$(date '+%H:%M:%S') cooldown query failed (rc=$RC), retrying..."
            echo "$OUTPUT" | tail -5
            sleep "$POLL"
            continue
        fi
        if echo "$OUTPUT" | grep -q "READY"; then
            echo "$(date '+%H:%M:%S') cooldown READY"
            break
        fi
        REMAINING="$(echo "$OUTPUT" | grep 'Remaining' | head -1 || echo '?')"
        echo "$(date '+%H:%M:%S') waiting ... $REMAINING"
        sleep "$POLL"
    done

    if [[ "$SAFETY" -gt 0 ]]; then
        echo "Sleeping ${SAFETY}s clock-skew safety..."
        sleep "$SAFETY"
    fi

    echo ""
    echo "=== Submitting at $(date) ==="
    set +e
    if [[ -n "$NOTES" ]]; then
        pixi run python track1_activity/scripts/api.py submit "$CSV" \
            --track "$TRACK" \
            --notes "$NOTES" \
            --experiment "$EXP"
        SUBMIT_RC=$?
    else
        pixi run python track1_activity/scripts/api.py submit "$CSV" \
            --track "$TRACK" \
            --experiment "$EXP"
        SUBMIT_RC=$?
    fi
    set -e

    echo ""
    echo "=== Post-submit status (rc=$SUBMIT_RC) ==="
    pixi run python track1_activity/scripts/api.py status --limit 3 || true

    if [[ $SUBMIT_RC -ne 0 ]]; then
        echo "SUBMIT FAILED (rc=$SUBMIT_RC)"
        exit 3
    fi

    echo ""
    echo "=== Done at $(date) ==="
} 2>&1 | tee "$LOG"
