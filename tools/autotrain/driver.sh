#!/usr/bin/env bash
# Autotrain loop (M5 plan, Phase B): waits on training runs and calls a fresh Claude agent for every decision.
#
# Run inside screen:  screen -S autotrain tools/autotrain/driver.sh
# Stop:               touch logs/autotrain/STOP   (the alive run finishes; no new one is launched)
#
#   no run alive          -> agent, mode `next`  (analyse the last run, change one thing, commit, launch)
#   run alive >= CHECK min -> agent, mode `check` (once per run: kill it only if clearly broken)
#   deadline or STOP file -> agent, mode `final` (Monday summary), then exit
# After every agent call the branch is pushed to origin (the agent itself may not push). Failed agent calls (e.g.
# usage limit) are retried after 30 min. The agent writes logs/autotrain/DONE to end the loop early.
#
# Settings (environment): AUTOTRAIN_DEADLINE ("2026-10-12 07:00"), AUTOTRAIN_BRANCH (exp/m5-weekend),
# AUTOTRAIN_CHECK_MIN (20), AUTOTRAIN_MODEL (sonnet), AUTOTRAIN_EFFORT (high), AUTOTRAIN_NOTE (extra line for the
# agent, e.g. for a dry run).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

DEADLINE=${AUTOTRAIN_DEADLINE:-"2026-10-12 07:00"}
BRANCH=${AUTOTRAIN_BRANCH:-exp/m5-weekend}
CHECK_MIN=${AUTOTRAIN_CHECK_MIN:-20}
MODEL=${AUTOTRAIN_MODEL:-sonnet}
EFFORT=${AUTOTRAIN_EFFORT:-high}
NOTE=${AUTOTRAIN_NOTE:-}
STATE=logs/autotrain
mkdir -p $STATE
export AUTOTRAIN_BRANCH=$BRANCH  # for launch_run.sh, called by the agent

log() { echo "$(date -Iseconds) $*" | tee -a $STATE/driver.log; }
run_pid() { [[ -f $STATE/current.pid ]] && cat $STATE/current.pid; }
alive() { local pid; pid=$(run_pid) && kill -0 "$pid" 2>/dev/null; }

call_agent() {  # $1 = mode
    local mode=$1 n attempt rc
    n=$(printf "%03d" $(( $(ls $STATE/agent_*.jsonl 2>/dev/null | wc -l) + 1 )))
    for attempt in 1 2 3 4; do
        log "agent $n mode=$mode attempt=$attempt"
        timeout 45m claude -p "$(cat tools/autotrain/prompt.md)

MODE: $mode
NOW: $(date '+%Y-%m-%d %H:%M')
DEADLINE: $DEADLINE
$NOTE" --model "$MODEL" --effort "$EFFORT" --settings tools/autotrain/settings.json --permission-mode dontAsk \
            --output-format stream-json --verbose > "$STATE/agent_${n}_${mode}_${attempt}.jsonl" 2>&1
        rc=$?
        (( rc == 0 )) && break
        log "agent $n failed (exit $rc); retry in 30 min"
        sleep 1800
    done
    [[ $(git rev-parse --abbrev-ref HEAD) == "$BRANCH" ]] || log "WARNING: not on $BRANCH after the agent call"
    git push -q origin "$BRANCH" 2>>$STATE/driver.log || log "push failed"
}

[[ $(git rev-parse --abbrev-ref HEAD) == "$BRANCH" ]] || { log "not on $BRANCH, exiting"; exit 1; }
deadline_s=$(date -d "$DEADLINE" +%s) || exit 1
log "driver start: deadline $DEADLINE, branch $BRANCH, model $MODEL ($EFFORT)"
idle_calls=0
while :; do
    if [[ -f $STATE/STOP ]] || (( $(date +%s) >= deadline_s )); then
        log "stop (STOP file or deadline)"
        call_agent final
        break
    fi
    if alive; then
        pid=$(run_pid)
        age_min=$(( ($(date +%s) - $(stat -c %Y $STATE/current.pid)) / 60 ))
        if [[ ! -f $STATE/checked.$pid ]] && (( age_min >= CHECK_MIN )); then
            touch $STATE/checked.$pid
            call_agent check
        else
            sleep 60
        fi
        continue
    fi
    [[ -f $STATE/DONE ]] && { log "agent wrote DONE"; break; }
    pid=$(run_pid) && kill -KILL -- "-$pid" 2>/dev/null && log "killed leftovers of run $pid"  # hung at shutdown
    call_agent next
    if alive; then
        idle_calls=0
    else
        idle_calls=$(( idle_calls + 1 ))
        log "agent launched no run ($idle_calls in a row)"
        (( idle_calls >= 3 )) && { log "3 calls without a run, exiting"; break; }
        sleep 600
    fi
done
log "driver end"
