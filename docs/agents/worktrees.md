# Worktrees

Any session that works on a branch while other sessions may use the main checkout (an `/implement` of one ticket, or an agent of an orchestrator, `orchestration.md`) works in its own git worktree under `.claude/worktrees/`: other sessions switch the main checkout's branch.

## Create

`git worktree add .claude/worktrees/<name> -b <branch> main`

## Setup

Before any `uv run` in the worktree:

1. `../IsaacLab` must resolve: the symlink `.claude/worktrees/IsaacLab -> /mnt/data/users/paolo/IsaacLab` exists.
   For an asset build, `../franka_description` resolves the same way: `.claude/worktrees/franka_description -> /mnt/data/users/paolo/franka_description`.
2. Generated assets are git-ignored. Link them from the main checkout:
   `ln -s /mnt/data/users/paolo/plant-stem-manipulation/assets/fr3/build assets/fr3/build`, same for `assets/stem/chain/build`.
   **Copy** instead of linking (`cp -a`) when the ticket rebuilds the assets, or when the user may rebuild the main checkout's assets while the session runs.
3. Own `.venv`: `uv sync --extra isaacsim --extra rsl-rl --extra wandb --extra video`. A `.venv` is never shared between worktrees (the editable install points at one tree).
4. Simulation runs need `OMNI_KIT_ACCEPT_EULA=YES`.

## GPU

Simulator runs (checks, recordings; not training runs) go through `scripts/gpu_slot.py`, with the system python:

`scripts/gpu_slot.py uv run --extra isaacsim python scripts/check_fr3.py`

It runs the command once a GPU slot is free (`flock` lock files shared by all worktrees; slot count and lock dir in the script), waits without polling, frees the slot when the command ends (also on a crash or kill) and returns the command's exit status. Never wait for a slot with a `pgrep` loop. Stagger launches by about a minute (shared shader cache).

## Merge

From the main checkout: remove the worktree first (`git worktree remove -f -f <path>`), then `gh pr merge N --squash`, delete the remote and local branch, `git pull --ff-only` on `main`.
Check the main checkout's branch with `git branch --show-current` before branch operations; the session-start git snapshot can be stale.
