"""Undecided values carry the one proposal tag `(proposal, YYYY-MM-DD, #N)`; no other proposal-like tag exists.

CODING_STANDARDS.md, "Decisions and history". The open proposals are listed with their issue as a warning, so the
orchestrator resolves them when the decision lands. Run with `-W default` or read the "warnings summary".
"""

import re
import subprocess
import warnings
from pathlib import Path

import pytest

pytestmark = pytest.mark.kitless

REPO_ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = (".py", ".yaml", ".yml")
SELF = "tests/test_decision_tags.py"

PROPOSAL_TAG = re.compile(r"\(proposal, (\d{4}-\d{2}-\d{2}), (#\d+)\)")
# Any other use of the word, or an orchestrator / pending-review tag. `proposal-original.md` is a file name.
BAD_TAG = re.compile(r"\bproposal\b(?!-original)|\(orchestrator|pending user review", re.IGNORECASE)


def tag_problems(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str, str]]]:
    """Return ((line, text) of each invalid tag, (line, date, issue) of each valid proposal tag) in ``text``."""
    bad, proposals = [], []
    for n, line in enumerate(text.splitlines(), start=1):
        rest = PROPOSAL_TAG.sub("", line)
        if BAD_TAG.search(rest):
            bad.append((n, line.strip()))
        proposals += [(n, *m.groups()) for m in PROPOSAL_TAG.finditer(line)]
    return bad, proposals


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [f for f in out if f.endswith(SUFFIXES) and f != SELF and (REPO_ROOT / f).is_file()]


@pytest.mark.parametrize(
    ("line", "valid"),
    [
        ("x = 1  # reset noise (proposal, 2026-10-15, #163)", True),
        ("x = 1  # reset noise (proposal 2026-10-15)", False),
        ("x = 1  # (proposal, 2026-10-15)", False),
        ("x = 1  # (orchestrator, 2026-10-15, pending user review)", False),
        ("x = 1  # a proposal", False),
        ("EXCLUDED = ('docs/proposal-original.md',)", True),
        ("x = 1  # (user, 2026-10-15)", True),
    ],
)
def test_tag_forms(line, valid):
    bad, _ = tag_problems(line)
    assert (not bad) == valid


def test_no_invalid_decision_tags():
    found = []
    for f in _tracked_files():
        bad, _ = tag_problems((REPO_ROOT / f).read_text())
        found += [f"{f}:{n}: {text}" for n, text in bad]
    assert not found, "Use `(proposal, YYYY-MM-DD, #N)` or a decided tag:\n" + "\n".join(found)


def test_list_open_proposals():
    open_ = []
    for f in _tracked_files():
        _, proposals = tag_problems((REPO_ROOT / f).read_text())
        open_ += [f"{issue} (since {date}): {f}:{n}" for n, date, issue in proposals]
    if open_:
        warnings.warn("Open proposals, resolve when decided:\n" + "\n".join(sorted(open_)), stacklevel=1)
