#!/bin/bash
# Daily run of the X market-signals pipeline (production, baseline arms and Dropbox uploads).
#
# Usage: run_daily_pipeline.sh [market_signals_x arguments]
#   With no arguments it runs: --baseline-arms --sync-dropbox
# Set PYTHON_BIN or MODEL to override the defaults on another machine.
# Each run writes cron_logs/run_<UTC time>.log under the data folder; latest.log
# points to the most recent one.

set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/market-signals-tiktok/bin/python}"
MODEL="${MODEL:-gpt-5-mini-2025-08-07}"
LOG_DIR="$REPO_DIR/ai_population/data/market-signals-x/cron_logs"
LOCK_FILE="$LOG_DIR/.running.pid"
LOG_RETENTION_DAYS=90

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

notify() {
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"$1\" with title \"Market signals pipeline\"" >/dev/null 2>&1 || true
    fi
}

mkdir -p "$LOG_DIR"

if [ -f "$LOCK_FILE" ] && kill -0 "$(cat "$LOCK_FILE")" 2>/dev/null; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') skipped: run $(cat "$LOCK_FILE") still in progress" >> "$LOG_DIR/skipped.log"
    notify "Skipped: the previous run is still in progress"
    exit 1
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

if [ "$#" -eq 0 ]; then
    set -- --baseline-arms --sync-dropbox
fi

LOG_FILE="$LOG_DIR/run_$(date -u '+%Y-%m-%dT%H%MZ').log"
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"

cd "$REPO_DIR" || exit 1
{
    echo "=== start $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname): $*"
    "$PYTHON_BIN" -m ai_population.src.market_signals_x --model "$MODEL" "$@"
    status=$?
    echo "=== end $(date -u '+%Y-%m-%d %H:%M:%S UTC') exit status $status"
} >> "$LOG_FILE" 2>&1

case "$status" in
    0) notify "Finished OK" ;;
    1) notify "Finished with WARN or an error (exit 1). See latest.log" ;;
    *) notify "FAILED (exit $status). See latest.log" ;;
esac

find "$LOG_DIR" -name 'run_*.log' -mtime +"$LOG_RETENTION_DAYS" -delete
exit "$status"
