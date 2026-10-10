#!/usr/bin/env bash
# One autotrain job, started by launch_run.sh: train (no cameras), then a play video of the last checkpoint
# (2 episodes, env 0, 4 envs), uploaded to the run's wandb page as `play_video`.
# Usage: tools/autotrain/run_job.sh <run name> [extra `isaaclab train` args]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN=$1
shift
STATE=logs/autotrain
TASK=StemManip-Push-Position-FR3-v0
VIDEO_STEPS=938  # 2 episodes of 469 steps (15 s at 31.25 Hz)

uv run --extra isaacsim isaaclab train --task $TASK --num_envs 4096 --run_name "$RUN" "$@" > "$STATE/$RUN.out" 2>&1
echo "train exit $?" >> "$STATE/$RUN.out"

RUN_DIR=$(ls -d logs/rsl_rl/stem_push_position/*_"$RUN" 2>/dev/null | tail -n 1)
CHECKPOINT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -n 1)
[[ -n $CHECKPOINT ]] || { echo "no checkpoint in '$RUN_DIR', no video" >> "$STATE/$RUN.play.out"; exit 0; }
timeout 20m uv run --extra isaacsim --extra video isaaclab play --task $TASK --num_envs 4 --checkpoint "$CHECKPOINT" \
    --video --video_length $VIDEO_STEPS > "$STATE/$RUN.play.out" 2>&1 &&
    uv run python tools/autotrain/upload_video.py "$STATE/$RUN.out" "$RUN_DIR" >> "$STATE/$RUN.play.out" 2>&1
echo "video exit $?" >> "$STATE/$RUN.play.out"
