#!/bin/bash
# Scheduled LB submission — polls api.py cooldown for the chosen track
# and submits once READY. Works for both Activity (CSV) and Structure
# (ZIP) submissions; the cooldown query is now track-filtered so it
# doesn't false-succeed when the OTHER track is READY.
#
# Designed to run unattended. Use `--daemon` to launch through `setsid` so it
# survives Codex app / non-interactive shell teardown. Logs to a timestamped file
# under logs/ for post-hoc audit.
#
# Usage:
#   scheduled_submit.sh <input_path> \
#       --experiment <experiment_name> \
#       [--notes <notes_text>] \
#       [--track <track_name>] \
#       [--poll <seconds>] \
#       [--safety <seconds>] \
#       [--log <log_path>] \
#       [--daemon]
#
#   <input_path> is a CSV for Activity Prediction or a ZIP for Structure
#   Prediction; api.py auto-detects from the file passed.
#
# Defaults:
#   --track   "Activity Prediction"     (use "Structure Prediction" for Track 2)
#   --poll    180   (check cooldown every 3 minutes)
#   --safety  60    (sleep this long after cooldown READY before submitting,
#                    in case of clock skew with LB server)
#   --log     logs/scheduled_submit_<timestamp>.log
#
# Typical daemon invocation (Track 1):
#   bash track1_activity/scripts/scheduled_submit.sh \
#       track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv \
#       --experiment ens_caruana_bag20_calibrated_best \
#       --notes "blah blah" \
#       --daemon
#
# Track 2 example:
#   bash track1_activity/scripts/scheduled_submit.sh \
#       track2_structure/submissions/track2_boltz2_zanchored_v2b_2026-04-26.zip \
#       --track "Structure Prediction" \
#       --experiment track2_zanchored_v2b \
#       --notes "Z-anchored ..." \
#       --daemon
#
# Exit codes:
#   0  submission successful
#   1  usage / argument error
#   2  input file not found
#   3  submission failed (api.py exit non-zero)
set -euo pipefail

CSV=""
EXP=""
NOTES=""
TRACK="Activity Prediction"
POLL=180
SAFETY=60
LOG=""
DAEMON=0
STATUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXP="$2"; shift 2 ;;
        --notes) NOTES="$2"; shift 2 ;;
        --track) TRACK="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        --safety) SAFETY="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --status-log) STATUS="$2"; shift 2 ;;
        --daemon) DAEMON=1; shift ;;
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
    echo "ERROR: input file not found: $CSV" >&2
    exit 2
fi

# Locate repo root (script is at <repo>/track1_activity/scripts/scheduled_submit.sh)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ -z "$LOG" ]]; then
    mkdir -p logs
    LOG="logs/scheduled_submit_$(date +%Y%m%d_%H%M%S).log"
fi

if [[ -z "$STATUS" ]]; then
    STATUS="/tmp/scheduled_submit_$(basename "$LOG" .log).status.log"
fi

if [[ "$DAEMON" -eq 1 ]]; then
    # Relaunch this script in a new session and return immediately. This is more
    # reliable than `nohup ... &` from Codex app / short-lived non-interactive
    # shells, where background children can be reaped as the shell exits.
    args=("$CSV" --experiment "$EXP" --track "$TRACK" --poll "$POLL" --safety "$SAFETY" --log "$LOG" --status-log "$STATUS")
    if [[ -n "$NOTES" ]]; then
        args+=(--notes "$NOTES")
    fi
    setsid bash "$0" "${args[@]}" </dev/null >"$STATUS" 2>&1 &
    PID=$!
    echo "scheduled_submit daemon started"
    echo "  pid:        $PID"
    echo "  log:        $LOG"
    echo "  status-log: $STATUS"
    echo "  stop:       pkill -f 'scheduled_submit.*$(basename "$CSV")'"
    exit 0
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

    # Map server-side track name to api.py cooldown short form.
    # Without filtering by track, the cooldown command shows both tracks
    # and READY/WAIT for either — the script would falsely succeed on
    # Track 2 submission when only Track 1 was READY (and vice versa).
    case "$TRACK" in
        "Activity Prediction") COOLDOWN_TRACK="activity" ;;
        "Structure Prediction") COOLDOWN_TRACK="structure" ;;
        *) echo "Unknown track: $TRACK" >&2; exit 1 ;;
    esac

    while true; do
        SET_PLUS_E=0
        set +e
        OUTPUT="$(pixi run python track1_activity/scripts/api.py cooldown --track "$COOLDOWN_TRACK" 2>&1)"
        RC=$?
        set -e
        if [[ $RC -ne 0 ]]; then
            echo "$(date '+%H:%M:%S') cooldown query failed (rc=$RC), retrying..."
            echo "$OUTPUT" | tail -5
            sleep "$POLL"
            continue
        fi
        if echo "$OUTPUT" | grep -q "READY"; then
            echo "$(date '+%H:%M:%S') cooldown READY ($COOLDOWN_TRACK)"
            break
        fi
        REMAINING="$(echo "$OUTPUT" | grep 'Remaining' | head -1 || echo '?')"
        echo "$(date '+%H:%M:%S') waiting [$COOLDOWN_TRACK] ... $REMAINING"
        sleep "$POLL"
    done

    if [[ "$SAFETY" -gt 0 ]]; then
        echo "Sleeping ${SAFETY}s clock-skew safety..."
        sleep "$SAFETY"
    fi

    # Pre-submit fetch: refresh the local lb_submissions DB with any
    # LB rows that have appeared since the previous submission. This
    # keeps the back-fill chain intact when a series of scheduled
    # submissions runs back-to-back: by the time the cooldown unlocks
    # and we're about to submit the next one, the previous submission's
    # LB row has likely been published, and pulling it now ensures the
    # local record is updated before we add a new pending row to the DB.
    echo ""
    echo "=== Pre-submit fetch (back-fill previous submission if any) ==="
    set +e
    pixi run python track1_activity/scripts/api.py fetch \
        --track "$COOLDOWN_TRACK" 2>&1 | tail -10
    set -e

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
    pixi run python track1_activity/scripts/api.py status \
        --limit 3 --track "$COOLDOWN_TRACK" || true

    if [[ $SUBMIT_RC -ne 0 ]]; then
        echo "SUBMIT FAILED (rc=$SUBMIT_RC)"
        exit 3
    fi

    echo ""
    echo "=== Done at $(date) ==="
} 2>&1 | tee "$LOG"
