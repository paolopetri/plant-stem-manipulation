# Workflow for new work

How work goes from an idea to a merged PR (user, 2026-10-14).

The user-only skills (`/to-spec`, `/to-tickets`, `/implement`, `/implement-spec`, `/handoff`, `/wayfinder`, `/improve-codebase-architecture`) are started by the user. When one fits the step, name it and let the user run it.

## Decide

1. `/grill-with-docs`: settle the decisions; `GLOSSARY.md` and ADRs are updated on the way.
2. `/to-spec`: one spec issue.
3. `/to-tickets`: tickets with blocking edges. The user approves the granularity and the edges before they are published.

## Implement

Pick by how much review each ticket needs:

| Case | How |
|---|---|
| One ticket, one session | `/implement #N` on the ticket's branch, then a PR. |
| Many tickets, little review each | `/implement-spec`: one integration branch, one `/code-review`, one PR. |
| Many tickets, each reviewed by the user (rules, docs, decisions) | An orchestrator session, one PR per ticket, as in #47. |

An agent-run ticket ends at an open PR; the merge needs the user's OK.

## Architecture

`/improve-codebase-architecture` -> grilling -> tickets.

## Context

- A session stays under ~120k tokens. Durable state lives in issues and repo files, never only in the chat.
- `/clear` between units of work; `/handoff` when the work continues in another session.

## Models

- Opus where judgement shapes text others rely on: rules, issues, docstrings, docs, audits.
- Sonnet where a tool verifies the result: ruff, CI, refactors covered by unit tests.
