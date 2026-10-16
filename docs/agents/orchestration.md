# Orchestrating agents

How an orchestrator session runs one background agent per ticket (learned in #47, user, 2026-10-14/15).

## Frontier

Open sub-issues of the parent whose blockers are all closed (`Blocked by` line or native dependencies). Start `ready-for-agent` tickets as agents with the model named in the ticket; work on `ready-for-human` tickets with the user.

## Worktrees

Agents running in parallel each get their own git worktree; a chain of agents (blocking edges) reuses one, with a new branch per ticket (`worktrees.md`, Create). Setup, the GPU limit and merging: `worktrees.md`. `gh` workarounds: `issue-tracker.md`.

## Context

- The orchestrator and every agent stay under ~120k tokens (`workflow.md`, Context). The spawn prompt states it: in the #160 run it reached the brief only late, and agents peaked at 266k.
- An agent that nears the budget stops at a clean point, writes its state (done, next step, open questions) to its scratchpad file and hands back with the path; the orchestrator continues with a fresh agent pointed at that file, as `/handoff` does between sessions.
- Read by section: `grep -n` for the place, then `sed -n 'a,bp'`. Never `cat` a large script, a whole spec with every comment (`scripts/read_issue.py N --comment K` reads one comment) or a whole diff.

## Agent prompt

- "Follow CLAUDE.md and `~/.claude/skills/implement/SKILL.md` on the issue's own branch, open a PR, do not merge."
- A refactor of simulator code records a fixed-seed rollout with `scripts/record_rollout.py` before the change, in the branch's worktree with the base commit checked out (not in the main checkout), and compares after it without `--tol`: bit-exact, except the contact-sensor force terms (raw contact force, contact-force reward) within their float32 noise floor.
- Each agent writes its PR body to its own scratchpad file (e.g. `pr-<N>.md`).
- Agents never create issues; new issues come from the user's `/to-spec` or `/to-tickets`.
- The spawn prompt itself names the issue-read command (`scripts/read_issue.py`, `issue-tracker.md`) and the sim check runner (`scripts/sim_check.py`, `worktrees.md` GPU). In #160 a warning only inside the brief came too late: plain `gh issue view` failed in about 8 agents.
- Before handing back, stop your own background tasks.

## Per PR

1. Check the agent's report against the `implement` steps and flag any skipped step.
2. Run `/code-review` against `main`; show the user the agent's summary and the findings.
3. Send only the findings the user approves back to the agent; record the decision on the ticket. The fix agent writes its red run (command and failing row) and every finding it declines, with the reason, into the PR body, not only into its handback: in #160 both stayed in the handback and the second review raised them again.
4. Merge after the user's OK and green CI (`gh pr checks N`), then comment one line on the parent issue.

## Merge into an integration branch

A ticket branch goes into the integration branch through `scripts/merge_ticket.py` (`worktrees.md`, Merge, step 1). Start an agent only when it exits with a `merge_ticket: FAILED (<cause>)` line; in #160 all 9 merger agents found the branch already up to date.

## Questions to the user

Each option states its effect in plain words: what changes in the code, the run or the result if the user picks it, one choice per option. Two #160 questions were rejected as unclear.

## End

Write the day's note (the orchestrator writes it, not the agents), update `docs/TODO.md`, close the parent; the user then runs `/retro`.
