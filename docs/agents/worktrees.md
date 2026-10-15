# Worktrees

Any session that works on a branch while other sessions may use the main checkout (an `/implement` of one ticket, or an agent of an orchestrator, `orchestration.md`) works in its own git worktree under `.claude/worktrees/`: other sessions switch the main checkout's branch.

## Create

`git worktree add .claude/worktrees/<name> -b <branch> main`

A new worktree costs about 35 min (`uv sync` with Isaac Sim, first Kit start). Create one per ticket that runs in parallel with others. Tickets that run one after another (a chain of blocking edges) reuse one set-up worktree: each ticket gets a new branch off the integration tip, once the previous ticket has landed there:
- `/implement-spec`: `git -C <path> switch -c <branch> origin/<integration branch>` once the previous ticket is pushed to it. `origin/<integration branch>`, not the local branch: the merge push ("Merge", step 2) moves only the remote branch and its tracking ref.
- Tickets with their own PR into `main`: `git -C <path> fetch origin main`, then `git -C <path> switch -c <branch> origin/main` once the previous PR is merged (user, 2026-10-15). Each squash PR then holds only its own ticket.

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

No step switches the main checkout's branch or pulls in it: other sessions may be using it.

1. Remove the worktree (`git worktree remove -f -f <path>`), unless the next ticket of a chain reuses it.
2. A PR: `gh pr merge N --squash`, then delete the remote and local branch.
   Without a PR (a ticket branch into an integration branch): fast-forward `<integration branch>` to `<ticket branch>` on the remote with `git push origin <ticket branch>:<integration branch>`, never by checking out `<integration branch>`.
   A reused worktree switches to the next ticket's branch before the local branch is deleted: git does not delete a branch checked out in a worktree.
3. Update local `main` with `git fetch origin main:main`. If git refuses (a worktree has `main` checked out, or local `main` has diverged), leave it: the session working on `main` pulls when it needs to.

Check the main checkout's branch with `git branch --show-current` before branch operations; the session-start git snapshot can be stale.
