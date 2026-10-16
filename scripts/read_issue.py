#!/usr/bin/env python3
"""Print a GitHub issue (title, body, comments) or one chosen comment (no simulator, standard library only).

Plain `gh issue view` fails here (Projects classic deprecation); this reads the issue through
`gh issue view --json`. `--comment K` prints only the K-th comment (1 = oldest), which saves context when one decision
sits in a long thread.

Usage: scripts/read_issue.py 160
       scripts/read_issue.py 162 --comment 2
Verify: tests/test_read_issue.py.
"""

import argparse
import json
import subprocess


def _format_comment(index: int, comment: dict) -> str:
    author = (comment.get("author") or {}).get("login", "unknown")
    return f"--- comment {index} by {author} at {comment['createdAt']} ---\n{comment['body']}"


def format_issue(issue: dict, comment: int | None = None) -> str:
    """Return the issue as text; with `comment` (1-based) only that comment."""
    comments = issue["comments"]
    if comment is not None:
        if not 1 <= comment <= len(comments):
            raise ValueError(f"#{issue['number']} has {len(comments)} comments, no comment {comment}")
        return f"#{issue['number']} {issue['title']}\n" + _format_comment(comment, comments[comment - 1])
    parts = [f"#{issue['number']} {issue['title']}", issue["body"]]
    parts += [_format_comment(i, c) for i, c in enumerate(comments, start=1)]
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("number", type=int, help="issue number")
    parser.add_argument("--comment", type=int, help="print only this comment (1 = oldest)")
    args = parser.parse_args()
    fields = "number,title,body,comments"
    out = subprocess.run(
        ["gh", "issue", "view", str(args.number), "--json", fields], capture_output=True, text=True, check=True
    ).stdout
    try:
        print(format_issue(json.loads(out), args.comment))
    except ValueError as err:
        raise SystemExit(f"read_issue: {err}") from None


if __name__ == "__main__":
    main()
