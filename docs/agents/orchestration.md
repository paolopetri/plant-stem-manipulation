# Orchestrating agents

How an orchestrator session runs one background agent per ticket (learned in #47, user, 2026-10-14/15).

## Frontier

Open sub-issues of the parent whose blockers are all closed (`Blocked by` line or native dependencies). Start `ready-for-agent` tickets as agents with the model named in the ticket; work on `ready-for-human` tickets with the user.

## Worktrees

Agents running in parallel each get their own git worktree; a chain of agents (blocking edges) reuses one, with a new branch per ticket (`worktrees.md`, Create). Setup, the GPU limit and merging: `worktrees.md`. `gh` workarounds: `issue-tracker.md`.

## Agent prompt

- "Follow CLAUDE.md and `~/.claude/skills/implement/SKILL.md` on the issue's own branch, open a PR, do not merge."
- A refactor of simulator code records a fixed-seed rollout with `scripts/record_rollout.py` before the change, in the branch's worktree with the base commit checked out (not in the main checkout), and compares after it without `--tol`: bit-exact, except the contact-sensor force terms (raw contact force, contact-force reward) within their float32 noise floor.
- Each agent writes its PR body to its own scratchpad file (e.g. `pr-<N>.md`).
- Agents never create issues; new issues come from the user's `/to-spec` or `/to-tickets`.
- Simulator runs go through `scripts/gpu_slot.py` (`worktrees.md`, GPU).
- Before handing back, stop your own background tasks.

## Per PR

1. Check the agent's report against the `implement` steps and flag any skipped step.
2. Run `/code-review` against `main`; show the user the agent's summary and the findings.
3. Send only the findings the user approves back to the agent; record the decision on the ticket.
4. Merge after the user's OK and green CI (`gh pr checks N`), then comment one line on the parent issue.

## End

Write the day's note (the orchestrator writes it, not the agents), update `docs/TODO.md`, close the parent; the user then runs `/retro`.
