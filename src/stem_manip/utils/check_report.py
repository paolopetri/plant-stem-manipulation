"""Shared verdict and exit status of the `scripts/check_*.py` (no simulator imports).

A check collects rows `{name: (ok, info)}`, ends with `finish`, which prints them, so CI and the
autotrain loop read PASS from exit status 0 and FAIL or NOT RUN from a non-zero one.
"""

from collections.abc import Collection, Mapping
from typing import NoReturn


def report(
    results: Mapping[str, tuple[bool, str]],
    not_run: Collection[str] = (),
    checked: Collection[str] | None = None,
) -> bool:
    """Print one `[PASS|FAIL|NOT RUN|INFO] name: info` line per row and the verdict; return True if all passed.

    Args:
        results: Rows `{name: (ok, info)}`.
        not_run: Names of rows that did not complete; they print NOT RUN and count as failed.
        checked: Names of rows that count (default all); the others print INFO.
    """
    counted = list(results) if checked is None else list(checked)
    for name, (ok, info) in results.items():
        if name in not_run:
            status = "NOT RUN"
        elif name not in counted:
            status = "INFO"
        else:
            status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {info}")
    passed = all(results[name][0] for name in counted) and not any(name in not_run for name in counted)
    print("=== all passed ===" if passed else "=== FAILED ===")
    return passed


def exit_status(passed: bool) -> int:
    """Return the process exit status of a check: 0 if it passed, 1 otherwise."""
    return 0 if passed else 1


def finish(
    results: Mapping[str, tuple[bool, str]],
    not_run: Collection[str] = (),
    checked: Collection[str] | None = None,
) -> NoReturn:
    """Print the rows and the verdict (see `report`) and exit with the check's status.

    Call it inside the `launch_simulation` block: the launcher passes the exit status on, whereas Kit would
    exit with 0 if the block ended normally.
    """
    raise SystemExit(exit_status(report(results, not_run, checked)))


def not_implemented(script: str, issue: str) -> NoReturn:
    """Exit non-zero with a message for a placeholder check; the spec lives in the issue."""
    raise SystemExit(f"{script}: not implemented (see {issue})")
