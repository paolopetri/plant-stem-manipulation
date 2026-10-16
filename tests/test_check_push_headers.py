"""Every `scripts/check_push_*.py` opens with a map of its phases (#178), so that an agent reads one section.

The map is the docstring block that starts with `Map (grep -n a marker`: one bullet `- <name>: `<marker>`` per phase
and a `New metric:` line. The tests check:

- Every check_push script has the map and the `New metric:` line.
- Each marker of the map occurs in the code after the docstring.
- Each `# -- ` phase comment of the code, and each name of `PHASES` in check_push_terms.py, is covered by a marker.
"""

import ast
import re
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("check_push_*.py"))
BULLET = re.compile(r"^- [^:]+: `(.+)`$")


def _split(script: Path) -> tuple[str, str]:
    """The module docstring and the code after it."""
    source = script.read_text()
    tree = ast.parse(source)
    doc = ast.get_docstring(tree, clean=False) or ""
    body_start = tree.body[1].lineno if len(tree.body) > 1 else len(source.splitlines())
    return doc, "\n".join(source.splitlines()[body_start - 1 :])


def _markers(doc: str) -> list[str]:
    lines = doc.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Map (grep -n a marker")), None)
    assert start is not None, "no `Map (grep -n a marker` block in the docstring"
    markers = []
    for line in lines[start + 1 :]:
        match = BULLET.match(line)
        if match:
            markers.append(match.group(1))
        elif not line.strip():
            break
    return markers


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_header_has_map_and_new_metric_line(script: Path):
    doc, _ = _split(script)
    assert _markers(doc), "the map lists no phase"
    assert any(line.startswith("New metric:") for line in doc.splitlines())


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_markers_occur_in_the_code(script: Path):
    doc, code = _split(script)
    missing = [m for m in _markers(doc) if m not in code]
    assert not missing, f"markers not in the code: {missing}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_phase_of_the_code_is_listed(script: Path):
    doc, code = _split(script)
    markers = _markers(doc)
    comments = [line.strip() for line in code.splitlines() if line.strip().startswith("# -- ")]
    unlisted = [c for c in comments if not any(c.startswith(m) for m in markers)]
    assert not unlisted, f"phase comments not in the map: {unlisted}"
    match = re.search(r"^PHASES = \((.*?)\)", code, re.M)
    if match:
        names = re.findall(r'"(\w+)"', match.group(1))
        unlisted = [n for n in names if f'phase == "{n}"' not in markers]
        assert not unlisted, f"PHASES not in the map: {unlisted}"
