#!/usr/bin/env python3
"""List the sim checks a commit range affects (no simulator, standard library only).

Before a merged head is pushed, its affected checks are re-run with `scripts/sim_check.py` (`docs/agents/worktrees.md`
-> Merge): a change on one ticket can break the checks of another (#179).

The mapping is a path table, first match wins. It is coarse on purpose: every push check builds the whole env, so a
change to any env term can break any of them. A path under `src/`, `assets/` or the dependencies with no finer rule
lists every check; anything else (docs, tests, other scripts) lists none. On top of the table, a changed check lists
itself, and a changed module (`src/` or a helper in `scripts/`) lists the checks that import it directly.

Usage: scripts/affected_checks.py origin/main..HEAD
       scripts/affected_checks.py "$(git merge-base HEAD^1 HEAD^2)..HEAD"   (after a merge commit)
Prints one check script per line (repo-relative), nothing if none. Verify: tests/test_affected_checks.py.
"""

import argparse
import re
import subprocess
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALL_CHECKS = sorted(f"scripts/{p.name}" for p in (REPO / "scripts").glob("check_*.py"))
PUSH = [c for c in ALL_CHECKS if c.startswith("scripts/check_push_")]
FR3 = ["scripts/check_fr3.py", "scripts/check_contact.py", *PUSH]
STEM = ["scripts/check_stem.py", "scripts/check_contact.py", *PUSH]

# (glob, checks); `*` also matches `/`.
RULES: list[tuple[str, list[str]]] = [
    ("src/stem_manip/tasks/push_position/agents/*", []),  # PPO cfg: training only
    ("src/stem_manip/tasks/*", PUSH),
    ("src/stem_manip/assets/fr3.py", FR3),
    ("assets/fr3/*", FR3),
    ("src/stem_manip/assets/stem/*", STEM),
    ("assets/stem/*", STEM),
    ("src/*", ALL_CHECKS),
    ("assets/*", ALL_CHECKS),
    ("pyproject.toml", ALL_CHECKS),
    ("uv.lock", ALL_CHECKS),
]


def _module(path: str) -> str | None:
    """The module a check imports for `path`: `scripts/x.py` -> `x`, `src/a/b.py` -> `a.b`, `src/a/__init__.py` -> `a`;
    None for any other path."""
    if not path.endswith(".py"):
        return None
    parts = Path(path).with_suffix("").parts
    if len(parts) == 2 and parts[0] == "scripts":
        return parts[1]
    if parts[0] == "src":
        return ".".join(parts[1:-1] if parts[-1] == "__init__" else parts[1:])
    return None


def importers(path: str) -> list[str]:
    """Check scripts importing the module at `path` directly (`import m`, `from m import`, `from pkg import m`)."""
    module = _module(path)
    if not module:
        return []
    package, _, name = module.rpartition(".")
    imports = rf"^\s*(import|from)\s+{re.escape(module)}\b"
    if package:
        imports += rf"|^\s*from\s+{re.escape(package)}\s+import\s+[^#\n]*\b{re.escape(name)}\b"
    pattern = re.compile(imports, re.MULTILINE)
    return [c for c in ALL_CHECKS if (REPO / c).is_file() and pattern.search((REPO / c).read_text())]


def affected_checks(paths: Iterable[str]) -> list[str]:
    """Return the sorted check scripts (repo-relative paths) that the changed `paths` affect."""
    checks: set[str] = set()
    for path in paths:
        checks.add(path)  # a changed check lists itself
        checks.update(importers(path))
        for pattern, hit in RULES:
            if fnmatch(path, pattern):
                checks.update(hit)
                break
    return sorted(checks & set(ALL_CHECKS))  # a deleted check script is not run


def changed_paths(commit_range: str) -> list[str]:
    """Paths changed in `commit_range` (both sides of a rename)."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", "--no-renames", commit_range],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.split()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("range", help="git commit range, e.g. origin/main..HEAD")
    args = parser.parse_args()
    for check in affected_checks(changed_paths(args.range)):
        print(check)


if __name__ == "__main__":
    main()
