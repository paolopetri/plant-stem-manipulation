# Worktrees

Any session that works on a branch while other sessions may use the main checkout (an `/implement` of one ticket, or an agent of an orchestrator, `orchestration.md`) works in its own git worktree under `.claude/worktrees/`: other sessions switch the main checkout's branch.

## Create

`git worktree add .claude/worktrees/<name> -b <branch> main`

A new worktree costs about 35 min (`uv sync` with Isaac Sim, first Kit start). Create one per ticket that runs in parallel with others. Tickets that run one after another (a chain of blocking edges) reuse one set-up worktree: each ticket gets a new branch off the integration tip, `git -C <path> switch -c <branch> <integration branch>`, once the previous ticket is committed.

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

At most two sessions run Isaac Sim at the same time (checks, not training runs); stagger their launches by about a minute (shared shader cache).

## Merge

No step switches the main checkout's branch or pulls in it: other sessions may be using it.

1. Remove the worktree (`git worktree remove -f -f <path>`), unless the next ticket of a chain reuses it.
2. A PR: `gh pr merge N --squash`, then delete the remote and local branch.
   A branch advanced without a PR (an integration branch by a ticket branch): fast-forward it on the remote with `git push origin <branch>:<target>`, never by checking out `<target>`.
3. Update local `main` with `git fetch origin main:main`. If git refuses because a tree has `main` checked out, leave it: the session in that tree pulls.

Check the main checkout's branch with `git branch --show-current` before branch operations; the session-start git snapshot can be stale.
