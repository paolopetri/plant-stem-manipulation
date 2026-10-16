#!/usr/bin/env python3
"""Merge the current ticket branch into an integration branch in one command (standard library only).

Run it in the ticket's worktree, with the system python. The steps (`docs/agents/worktrees.md` -> Merge):

1. Fetch; if the branch lacks the integration tip, merge `origin/<integration>` into it (a conflict is aborted).
2. pytest and ruff.
3. The sim checks the push brings changes to (`affected_checks.py` on `origin/<integration>..HEAD`), each through
   `sim_check.py`. A change that touches no sim code runs none.
4. Fast-forward push of HEAD to `<integration>` (never forced).
5. The one-line comment on the ticket: "Merged into <integration> at <sha>."

It stops at the first failure with a non-zero exit and a `merge_ticket: FAILED (<cause>)` line, cause one of
dirty worktree, conflict, tests, sim check, non-fast-forward, push, comment. `--dry-run` does every step but pushes
with `git push --dry-run` (which still rejects a non-fast-forward) and prints the comment instead of posting it.

Usage: scripts/merge_ticket.py feat/173-agent-environment 180 [--dry-run]
Verify: tests/test_merge_ticket.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from affected_checks import affected_checks, changed_paths

HERE = Path(__file__).resolve().parent
TESTS = [
    ["uv", "run", "pytest", "-q"],
    ["uv", "run", "ruff", "check", "."],
    ["uv", "run", "ruff", "format", "--check", "."],
]
SIM_CHECK = [sys.executable, str(HERE / "sim_check.py"), "uv", "run", "--extra", "isaacsim", "python"]  # + check
GH_COMMENT = ["gh", "issue", "comment"]  # + issue, --body, text
TAIL = 20  # output lines shown of a failed command


class Failed(Exception):
    """A step failed; the message names the cause and what to look at."""

    def __init__(self, cause: str, detail: str) -> None:
        super().__init__(f"merge_ticket: FAILED ({cause}): {detail}")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git in the current directory; output captured."""
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def _tail(text: str) -> str:
    return "\n".join(text.rstrip().splitlines()[-TAIL:])


def merge_tip(integration: str) -> None:
    """Fetch and make sure HEAD contains the integration tip, merging it in if not."""
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Failed("dirty worktree", "commit or stash the changes first")
    git("fetch", "-q", "origin")
    tip = f"origin/{integration}"
    if git("merge-base", "--is-ancestor", tip, "HEAD", check=False).returncode == 0:
        print(f"merge_ticket: the branch has the tip of {tip}", flush=True)
        return
    if git("merge", "--no-edit", tip, check=False).returncode != 0:
        files = git("diff", "--name-only", "--diff-filter=U").stdout.split()
        git("merge", "--abort", check=False)
        raise Failed("conflict", f"merging {tip} conflicts in {', '.join(files) or '?'}; merge by hand, then rerun")
    print(f"merge_ticket: merged {tip} at {git('rev-parse', '--short', 'HEAD').stdout.strip()}", flush=True)


def run_tests() -> None:
    """pytest and ruff; on a failure show the tail of its output."""
    for command in TESTS:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(_tail(result.stdout + result.stderr), flush=True)
            raise Failed("tests", f"`{' '.join(command)}` exited {result.returncode}")
    print("merge_ticket: tests passed", flush=True)


def run_sim_checks(integration: str) -> list[str]:
    """Run the sim checks affected by `origin/<integration>..HEAD` through the runner; return them."""
    checks = affected_checks(changed_paths(f"origin/{integration}..HEAD"))
    print(f"merge_ticket: sim checks: {', '.join(checks) or 'none'}", flush=True)
    for check in checks:
        status = subprocess.run([*SIM_CHECK, check]).returncode  # the runner prints its own filtered lines
        if status != 0:
            raise Failed("sim check", f"{check} exited {status}; its log path is in the runner's first line")
    return checks


def push(integration: str, dry_run: bool) -> None:
    """Fast-forward `<integration>` on origin to HEAD; a non-fast-forward is rejected, never forced."""
    args = ["push", *(["--dry-run"] if dry_run else []), "origin", f"HEAD:refs/heads/{integration}"]
    result = git(*args, check=False)
    if result.returncode != 0:
        rejected = "[rejected]" in result.stderr or "non-fast-forward" in result.stderr
        if rejected:
            raise Failed("non-fast-forward", f"{integration} moved on origin, rerun\n{_tail(result.stderr)}")
        raise Failed("push", _tail(result.stderr))
    print(f"merge_ticket: {'dry run: would push' if dry_run else 'pushed'} HEAD to {integration}", flush=True)


def comment(issue: str, integration: str, dry_run: bool) -> None:
    """The one-line comment on the ticket."""
    text = f"Merged into {integration} at {git('rev-parse', '--short', 'HEAD').stdout.strip()}."
    if dry_run:
        print(f"merge_ticket: dry run: would comment on #{issue}: {text}", flush=True)
        return
    result = subprocess.run([*GH_COMMENT, issue, "--body", text], capture_output=True, text=True)
    if result.returncode != 0:
        raise Failed("comment", f"pushed, but the comment on #{issue} failed: {_tail(result.stderr)}")
    print(f"merge_ticket: commented on #{issue}: {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("integration", help="integration branch, e.g. feat/173-agent-environment")
    parser.add_argument("issue", help="ticket issue number, for the comment")
    parser.add_argument("--dry-run", action="store_true", help="every step, but no push and no comment")
    args = parser.parse_args(argv)
    try:
        merge_tip(args.integration)
        run_tests()
        run_sim_checks(args.integration)
        push(args.integration, args.dry_run)
        comment(args.issue, args.integration, args.dry_run)
    except Failed as e:
        print(e, flush=True)
        return 1
    print("merge_ticket: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
