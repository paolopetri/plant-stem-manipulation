#!/usr/bin/env bash
# Launch one training run of the autotrain loop in the background (M5 plan, Phase B).
#
# Usage: tools/autotrain/launch_run.sh <NN> <short_name> "<hypothesis>" [extra `isaaclab train` args]
#   e.g. tools/autotrain/launch_run.sh 02 noise15 "Init std 1.5 explores more" --max_iterations 1500
#
# Refuses unless the tree is clean and on the autotrain branch, so every run is an exact commit, and unless no run
# is alive (one run at a time on the GPU). The job (run_job.sh: train with 4096 envs unless overridden, then a play
# video uploaded to wandb) gets its own process group (stop it with stop_run.sh) and is killed after
# AUTOTRAIN_MAX_HOURS. Writes logs/autotrain/current.pid and a line in logs/autotrain/runs.log.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BRANCH=${AUTOTRAIN_BRANCH:-exp/m5-weekend}
GROUP=${AUTOTRAIN_GROUP:-m5-weekend}
MAX_HOURS=${AUTOTRAIN_MAX_HOURS:-4}
STATE=logs/autotrain

fail() { echo "launch_run: $*" >&2; exit 1; }

[[ $# -ge 3 ]] || fail "usage: launch_run.sh <NN> <short_name> \"<hypothesis>\" [extra train args]"
NN=$1 NAME=$2 HYPOTHESIS=$3
shift 3
[[ $NN =~ ^[0-9]{2}$ ]] || fail "run number must be two digits, got '$NN'"
[[ $NAME =~ ^[a-z0-9_]+$ ]] || fail "short name must be [a-z0-9_]+, got '$NAME'"
[[ $(git rev-parse --abbrev-ref HEAD) == "$BRANCH" ]] || fail "not on $BRANCH"
[[ -z $(git status --porcelain) ]] || fail "uncommitted changes: commit them first (the run must be an exact commit)"
if [[ -f $STATE/current.pid ]] && kill -0 "$(cat $STATE/current.pid)" 2>/dev/null; then
    fail "a run is still alive (pid $(cat $STATE/current.pid))"
fi

mkdir -p $STATE
RUN=r${NN}_${NAME}
export WANDB_RUN_GROUP=$GROUP WANDB_NOTES=$HYPOTHESIS
setsid timeout "${MAX_HOURS}h" tools/autotrain/run_job.sh "$RUN" "$@" > /dev/null 2>&1 < /dev/null &
PID=$!
echo $PID > $STATE/current.pid
echo "$(date -Iseconds) $RUN commit=$(git rev-parse --short=12 HEAD) pid=$PID args=$* hypothesis=$HYPOTHESIS" \
    >> $STATE/runs.log
echo "launched $RUN (pid $PID); output $STATE/$RUN.out; run dir logs/rsl_rl/stem_push_position/<date>_$RUN"
