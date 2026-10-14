"""Unit tests for `stem_manip.utils.check_report`: the shared verdict and exit status of the `scripts/check_*.py`.
The tests check:

- All rows pass: verdict "all passed", exit status 0.
- `finish` raises SystemExit with that status (inside `launch_simulation` the launcher passes it on).
- Any FAIL row, or any NOT RUN row: verdict "FAILED", non-zero exit status.
- A NOT RUN row prints as NOT RUN even if its value was ok; rows outside `checked` print as INFO and do not count.
"""

import pytest

from stem_manip.utils import check_report


def test_all_passed_exits_zero(capsys):
    results = {"a": (True, "x"), "b": (True, "y")}
    assert check_report.report(results) is True
    out = capsys.readouterr().out
    assert "[PASS] a: x" in out and "[PASS] b: y" in out
    assert "=== all passed ===" in out
    assert check_report.exit_status(True) == 0


def test_fail_row_fails(capsys):
    assert check_report.report({"a": (True, "x"), "b": (False, "y")}) is False
    out = capsys.readouterr().out
    assert "[FAIL] b: y" in out and "=== FAILED ===" in out
    assert check_report.exit_status(False) != 0


def test_not_run_fails_and_prints_not_run(capsys):
    assert check_report.report({"a": (True, "x")}, not_run={"a"}) is False
    out = capsys.readouterr().out
    assert "[NOT RUN] a: x" in out and "=== FAILED ===" in out


def test_unchecked_rows_are_info_and_do_not_count(capsys):
    results = {"a": (True, "x"), "b": (False, "y")}
    assert check_report.report(results, checked=["a"]) is True
    out = capsys.readouterr().out
    assert "[INFO] b: y" in out and "=== all passed ===" in out


def test_finish_exits_with_the_verdict():
    with pytest.raises(SystemExit) as ok:
        check_report.finish({"a": (True, "x")})
    assert ok.value.code == 0
    with pytest.raises(SystemExit) as bad:
        check_report.finish({"a": (True, "x"), "b": (False, "y")})
    assert bad.value.code != 0
    with pytest.raises(SystemExit) as not_run:
        check_report.finish({"a": (True, "x")}, not_run={"a"})
    assert not_run.value.code != 0
