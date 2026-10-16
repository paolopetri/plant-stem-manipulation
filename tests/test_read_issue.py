"""`scripts/read_issue.py` prints an issue, or one chosen comment of it, from the `gh issue view --json` data (#176)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "read_issue.py"
spec = importlib.util.spec_from_file_location("read_issue", SCRIPT)
read_issue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read_issue)

ISSUE = {
    "number": 7,
    "title": "A title",
    "body": "The body.",
    "comments": [
        {"author": {"login": "alice"}, "createdAt": "2026-10-01T10:00:00Z", "body": "first comment"},
        {"author": {"login": "bob"}, "createdAt": "2026-10-02T11:00:00Z", "body": "second comment"},
    ],
}


def test_whole_issue_has_title_body_and_all_comments():
    text = read_issue.format_issue(ISSUE)
    assert "#7" in text and "A title" in text and "The body." in text
    assert "first comment" in text and "second comment" in text
    assert "alice" in text and "bob" in text


def test_chosen_comment_only():
    text = read_issue.format_issue(ISSUE, comment=2)
    assert "second comment" in text and "bob" in text
    assert "first comment" not in text
    assert "The body." not in text


def test_comment_out_of_range_names_the_count():
    with pytest.raises(ValueError, match="2 comments"):
        read_issue.format_issue(ISSUE, comment=3)
    with pytest.raises(ValueError, match="2 comments"):
        read_issue.format_issue(ISSUE, comment=0)
