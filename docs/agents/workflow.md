# Workflow for new work

How work goes from an idea to a merged PR (user, 2026-10-14).

The user-only skills (`/to-spec`, `/to-tickets`, `/implement`, `/implement-spec`, `/handoff`, `/wayfinder`, `/improve-codebase-architecture`, `/retro`) are started by the user. When one fits the step, name it and let the user run it.

## Decide

1. `/grill-with-docs`: settle the decisions; `GLOSSARY.md` and ADRs are updated on the way.
2. `/to-spec`: one spec issue.
3. `/to-tickets`: tickets with blocking edges. The user approves the granularity and the edges before they are published.

## Implement

Agent-run tickets are the way of working: the agent implements, commits and opens the PR on its own, and stops at the open PR. The merge needs the user's OK. Pick the way by how much review each ticket needs:

| Case | How |
|---|---|
| One ticket, one session | `/implement #N` on the ticket's branch (`/code-review` at the end), then a PR. In a worktree (`worktrees.md`) when other sessions may use the main checkout. |
| Many tickets, little review each | `/implement-spec`: one integration branch, one `/code-review`, one PR. |
| Many tickets, each reviewed by the user (rules, docs, decisions) | An orchestrator session starts one agent per ticket; each opens a PR, the orchestrator runs `/code-review` on it. Setup and routine: `docs/agents/orchestration.md`. |

PR bodies follow the `pr` skill (`.github/pull_request_template.md`).

## Retro

After a spec or an orchestrated parent issue is merged, the user runs `/retro` on its sessions (user, 2026-10-15). Its suggestions are applied only with the user's OK: as checks or `CODING_STANDARDS.md` rules, or as tickets via `/to-tickets`.

## Architecture

`/improve-codebase-architecture` -> grilling -> tickets.

## Context

- A session stays under ~120k tokens. Durable state lives in issues and repo files, never only in the chat.
- `/clear` between units of work; `/handoff` when the work continues in another session.

## Models

- Opus where judgement shapes text others rely on: rules, issues, docstrings, docs, audits.
- Sonnet where a tool verifies the result: ruff, CI, refactors covered by unit tests.
