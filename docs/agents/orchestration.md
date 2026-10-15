# Orchestrating agents

How an orchestrator session runs one background agent per ticket (learned in #47, user, 2026-10-14/15).

## Frontier

Open sub-issues of the parent whose blockers are all closed (`Blocked by` line or native dependencies). Start `ready-for-agent` tickets as agents with the model named in the ticket; work on `ready-for-human` tickets with the user.

## Worktrees

Each agent gets its own git worktree: `worktrees.md`.

## GPU

At most two agents run Isaac Sim at the same time (checks, not training runs); stagger their launches by about a minute (shared shader cache).

## Agent prompt

- "Follow CLAUDE.md and `~/.claude/skills/implement/SKILL.md` on the issue's own branch, open a PR, do not merge."
- A refactor of simulator code records a fixed-seed rollout with `scripts/record_rollout.py` before the change and compares at tol 0 after it.
- Each agent writes its PR body to its own scratchpad file (e.g. `pr-<N>.md`).
- Agents never create issues; new issues come from the user's `/to-spec` or `/to-tickets`.

## Per PR

1. Check the agent's report against the `implement` steps and flag any skipped step.
2. Run `/code-review` against `main`; show the user the agent's summary and the findings.
3. Send only the findings the user approves back to the agent; record the decision on the ticket.
4. Merge after the user's OK and green CI (below), then comment one line on the parent issue.

## gh and git

- Issue and PR views: `gh issue view N --json title,body,comments`; plain `gh issue view` and `gh pr edit` fail (Projects classic). Set PR bodies with `gh api -X PATCH repos/paolopetri/plant-stem-manipulation/pulls/N -F body=@<file>`.
- Merge: remove the worktree first (`worktrees.md`), then `gh pr merge N --squash`, delete the remote and local branch, `git pull --ff-only` on `main`.
- Check the main checkout's branch with `git branch --show-current` before branch operations; the session-start git snapshot can be stale.

## End

Write the day's note (the orchestrator writes it, not the agents), update `docs/TODO.md`, close the parent; the user then runs `/retro`.
