#!/bin/bash
# Daily run of the X market-signals pipeline (production, baseline arms and Dropbox uploads).
#
# Usage: run_daily_pipeline.sh [market_signals_x arguments]
#   With no arguments it runs: --baseline-arms --sync-dropbox, at most once per day
#   (Singapore date), so cron can call it every 15 minutes and missed slots are
#   caught up. Daily runs only start from 08:00 Singapore time, when the drop date
#   rolls over. Set FORCE_RUN=1 to run again on a day that has already started,
#   or before 08:00.
# Set PYTHON_BIN or MODEL to override the defaults on another machine.
# Each run writes cron_logs/run_<UTC time>.log under the data folder; latest.log
# points to the most recent one.

set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/market-signals-tiktok/bin/python}"
MODEL="${MODEL:-gpt-5-mini-2025-08-07}"
FORCE_RUN="${FORCE_RUN:-0}"
LOG_DIR="$REPO_DIR/ai_population/data/market-signals-x/cron_logs"
LOCK_FILE="$LOG_DIR/.running.pid"
STARTED_MARKER="$LOG_DIR/.started_$(TZ=Asia/Singapore date '+%Y-%m-%d')"
LOG_RETENTION_DAYS=90

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

notify() {
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"$1\" with title \"Market signals pipeline\"" >/dev/null 2>&1 || true
    fi
}

# On macOS, keep the machine from sleeping while the pipeline runs (idle sleep on
# battery or AC, system sleep on AC); closing the lid still sleeps it.
keep_awake() {
    if command -v caffeinate >/dev/null 2>&1; then
        caffeinate -is "$@"
    else
        "$@"
    fi
}

mkdir -p "$LOG_DIR"

DAILY_RUN=0
if [ "$#" -eq 0 ]; then
    DAILY_RUN=1
    if [ "$FORCE_RUN" != "1" ]; then
        if [ -f "$STARTED_MARKER" ] || [ "$(TZ=Asia/Singapore date '+%H')" -lt 8 ]; then
            exit 0
        fi
    fi
fi

if [ -f "$LOCK_FILE" ] && kill -0 "$(cat "$LOCK_FILE")" 2>/dev/null; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') skipped: run $(cat "$LOCK_FILE") still in progress" >> "$LOG_DIR/skipped.log"
    exit 0
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

if [ "$DAILY_RUN" -eq 1 ]; then
    touch "$STARTED_MARKER"
    set -- --baseline-arms --sync-dropbox
fi

LOG_FILE="$LOG_DIR/run_$(date -u '+%Y-%m-%dT%H%MZ').log"
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"

cd "$REPO_DIR" || exit 1
{
    echo "=== start $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname): $*"
    if command -v pmset >/dev/null 2>&1; then
        echo "=== power: $(pmset -g batt | head -1); sleep prevented while running"
    fi
    keep_awake "$PYTHON_BIN" -m ai_population.src.market_signals_x --model "$MODEL" "$@"
    status=$?
    echo "=== end $(date -u '+%Y-%m-%d %H:%M:%S UTC') exit status $status"
} >> "$LOG_FILE" 2>&1

case "$status" in
    0) notify "Finished OK" ;;
    1) notify "Finished with WARN or an error (exit 1). See latest.log" ;;
    *) notify "FAILED (exit $status). See latest.log" ;;
esac

find "$LOG_DIR" \( -name 'run_*.log' -o -name '.started_*' \) -mtime +"$LOG_RETENTION_DAYS" -delete
exit "$status"
