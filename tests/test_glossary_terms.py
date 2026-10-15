"""No tracked file uses an unambiguous _Avoid_ term from GLOSSARY.md.

The term list is an independent copy of the glossary (CODING_STANDARDS.md, "Single source of truth"). Ambiguous words
(tip, tool, target, goal) stay a review judgement and are not listed.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoided term -> glossary term to use instead.
AVOIDED_TERMS = {
    "tool tip": "slot base",
    "TCP": "slot base",
    "fork tip": "slot base",
    "soft fraction": "curvature budget",
    "soft limit": "curvature budget",
    "target offset": "setpoint offset",
    "impedance target": "setpoint",
}

EXCLUDED = (
    "docs/notes/",
    "docs/experiments/",
    "docs/proposal-original.md",
    "GLOSSARY.md",
    "tests/test_glossary_terms.py",
)


def _pattern(term: str) -> re.Pattern:
    """Match ``term`` in any case, with its words joined by space, ``_``, ``-`` or nothing (``ToolTip``)."""
    words = r"[ _-]?".join(re.escape(w) for w in term.split())
    # A single word must not sit inside a longer word ("tcpdump"); ``_`` and digits still delimit it ("tcp_pos").
    if " " not in term:
        words = rf"(?<![a-z]){words}(?![a-z])"
    return re.compile(words, re.IGNORECASE)


_PATTERNS = {term: _pattern(term) for term in AVOIDED_TERMS}


def avoided_terms_in(text: str) -> list[tuple[int, str, str]]:
    """Return (line number, avoided term, term to use) for each avoided term in ``text``."""
    return [
        (n, term, AVOIDED_TERMS[term])
        for n, line in enumerate(text.splitlines(), start=1)
        for term, pattern in _PATTERNS.items()
        if pattern.search(line)
    ]


@pytest.mark.parametrize(
    ("line", "term"),
    [
        ("the tool tip pose", "tool tip"),
        ("self.tool_tip_pos = x", "tool tip"),
        ("class ToolTipFrame:", "tool tip"),
        ("ee_fork-tip offset", "fork tip"),
        ("tcp_pos_w", "TCP"),
        ("the TCP frame", "TCP"),
        ("SOFT_FRACTION = 0.8", "soft fraction"),
        ("soft limit of the curvature", "soft limit"),
        ("targetOffset", "target offset"),
        ("impedance_target", "impedance target"),
    ],
)
def test_matcher_finds_avoided_term_in_any_spelling(line, term):
    assert [(n, t) for n, t, _ in avoided_terms_in(f"ok\n{line}\n")] == [(2, term)]


@pytest.mark.parametrize(
    "line", ["the slot base pose", "setpoint offset", "curvature budget", "tcpdump", "the tip and the tool"]
)
def test_matcher_ignores_allowed_words(line):
    assert avoided_terms_in(line) == []


def test_matcher_names_the_term_to_use():
    assert avoided_terms_in("soft_fraction") == [(1, "soft fraction", "curvature budget")]


def test_no_avoided_glossary_term_in_tracked_files():
    files = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    hits = []
    for rel in files.splitlines():
        if rel.startswith(EXCLUDED):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary files (meshes, images), deleted or submodule paths
        hits += [f"{rel}:{n}: '{term}', use '{use}'" for n, term, use in avoided_terms_in(text)]
    assert not hits, "Avoided glossary terms (GLOSSARY.md):\n" + "\n".join(hits)
