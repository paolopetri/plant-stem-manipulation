#!/usr/bin/env bash
# Stop the alive autotrain run: SIGTERM to its whole process group (uv, Isaac Sim), SIGKILL after 30 s.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PIDFILE=logs/autotrain/current.pid
[[ -f $PIDFILE ]] || { echo "stop_run: no run recorded"; exit 0; }
PID=$(cat $PIDFILE)
kill -0 "$PID" 2>/dev/null || { echo "stop_run: run $PID is not alive"; exit 0; }
kill -TERM -- "-$PID"
for _ in $(seq 30); do
    kill -0 "$PID" 2>/dev/null || { echo "stop_run: stopped $PID"; exit 0; }
    sleep 1
done
kill -KILL -- "-$PID" 2>/dev/null
echo "stop_run: killed $PID"
