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

## Merge

Remove the worktree first (`git worktree remove -f -f <path>`, from the main checkout), then merge the PR as usual.
